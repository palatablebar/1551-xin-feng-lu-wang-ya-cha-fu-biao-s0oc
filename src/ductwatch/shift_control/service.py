from dataclasses import replace
from statistics import median

from ..building_cohorts.service import zone_lookup, columns, central_gap
from ..fan_replay.service import check_point
from ..flow_subspace.service import predict_one
from ..ventilation_factors.contracts import FEATURE_KEYS, FEATURE_NAMES
from .contracts import CohortShift, ShiftScope, ShiftDecision, ShiftControlReport

# 小样本当班分区的 CDF 差天然偏高，0.62 仅在整区迁移或未知区时触发。
COHORT_GAP_LIMIT = 0.62
# 重算放行要求该作业区校准窗口至少有可对照实绩。
COVERAGE_FLOOR = 5
# 位置校正只吸收温和偏移；位移过大说明参考域失效，转冻结。
DRIFT_DELTA_LIMIT = 2.0

FEATURE_TITLE = dict(zip(FEATURE_KEYS, FEATURE_NAMES, strict=True))


def cohort_drift(reference, observed, calibration):
    """逐作业区计算最大分区累积分布差与实绩覆盖。"""
    groups = zone_lookup(observed)
    calibration_count = {}
    for row in calibration:
        calibration_count[row.cohort] = calibration_count.get(row.cohort, 0) + 1
    studies = []
    for cohort in sorted(groups):
        units = groups[cohort]
        history_bands = reference.cohorts.get(cohort)
        novel = history_bands is None
        history = reference.bands if novel else history_bands
        train_n = len(reference.cohorts.get(cohort, ()))
        gaps = []
        for name, old_series, new_series in zip(
                FEATURE_KEYS, history, columns(units), strict=True):
            gaps.append((name, central_gap(old_series, new_series)))
        driver, gap = max(gaps, key=lambda item: item[1])
        studies.append(CohortShift(
            cohort=cohort,
            live_samples=len(units),
            train_samples=train_n,
            calibration_samples=calibration_count.get(cohort, 0),
            max_gap=round(gap, 6),
            driver=FEATURE_TITLE[driver],
            shifted=novel or gap > COHORT_GAP_LIMIT,
            novel=novel,
        ))
    return tuple(studies)


def unit_flags(observed):
    """单条单据层面的偏移证据：两项缺测或两项超出训练包络。"""
    reasons = {}
    for vector in observed:
        flags = []
        if len(vector.missing) >= 2:
            flags.append('UNIT_MEASUREMENT_GAP')
        if len(vector.excursions) >= 2:
            flags.append('UNIT_ENVELOPE_BREACH')
        if flags:
            reasons[vector.key] = tuple(flags)
    return reasons


def define_scope(studies, observed):
    shifted_cohorts = tuple(item.cohort for item in studies if item.shifted)
    reasons = unit_flags(observed)
    scoped = frozenset(reasons) | frozenset(
        vector.key for vector in observed if vector.cohort in shifted_cohorts)
    return ShiftScope(tuple(sorted(shifted_cohorts)), tuple(sorted(scoped)), dict(sorted(reasons.items())))


def freeze_reason(vector, study):
    """返回冻结原因；返回 None 表示满足重算放行条件。"""
    if study is None or study.novel:
        return 'NO_COHORT_REFERENCE', '作业区未登记训练参考，域外评分不能重算'
    if len(vector.missing) >= 2:
        return 'MEASUREMENT_GAP', '两项及以上量测缺测，补值不足以支持重算'
    if len(vector.excursions) >= 2:
        return 'ENVELOPE_BREACH', '两项及以上特征超出训练包络，参考域不覆盖该工况'
    if study.calibration_samples < COVERAGE_FLOOR:
        return 'COVERAGE_INSUFFICIENT', f'校准实绩 {study.calibration_samples} 条，低于放行下限 {COVERAGE_FLOOR}'
    return None


def cohort_centering(reference_vectors, live_vectors):
    """按作业区计算训练→当班的标准化特征中位位移。"""
    train_groups = zone_lookup(reference_vectors)
    live_groups = zone_lookup(live_vectors)
    centering = {}
    for cohort, units in live_groups.items():
        history = train_groups.get(cohort)
        if not history:
            continue
        train_columns = columns(history)
        live_columns = columns(units)
        centering[cohort] = tuple(
            median(live) - median(old)
            for old, live in zip(train_columns, live_columns, strict=True))
    return centering


def recalculate(vector, delta, kernel, feature_stamp, verification):
    """在冻结的特征、PCA 模型与校准上做位置校正后重新评分。"""
    corrected = tuple(
        min(12.0, max(-12.0, value - shift))
        for value, shift in zip(vector.values, delta, strict=True))
    adjusted = replace(vector, values=corrected)
    prediction = predict_one(kernel, adjusted)
    assessment = check_point(prediction, verification)
    return round(assessment.estimate, 4), tuple(round(part, 4) for part in delta)


def evaluate_shift_control(reference_profile, observed, calibration, kernel,
                           feature_stamp, verification, reference_vectors,
                           live_vectors, plan):
    """生成偏移边界，并对范围内已派出单据给出冻结或重算放行结论。"""
    if feature_stamp != reference_profile.signature:
        raise ValueError('工区版本发生变化，需重建基准')
    studies = cohort_drift(reference_profile, observed, calibration)
    scope = define_scope(studies, observed)
    study_by_cohort = {item.cohort: item for item in studies}
    vector_by_key = {item.key: item for item in observed}
    deltas = cohort_centering(reference_vectors, live_vectors)

    dispatched = tuple(job for job in plan.jobs if job.state == '已分派')
    decisions = []
    for job in dispatched:
        if not scope.touches(job.key, job.cohort):
            continue
        vector = vector_by_key[job.key]
        study = study_by_cohort.get(job.cohort)
        blocking = freeze_reason(vector, study)
        evidence = (
            ('作业区', job.cohort),
            ('最大分区累积分布差', None if study is None else study.max_gap),
            ('偏移主导维度', None if study is None else study.driver),
            ('校准实绩数', 0 if study is None else study.calibration_samples),
            ('缺测项', vector.missing),
            ('超包络项', vector.excursions),
            ('单元级证据', scope.unit_reasons.get(job.key, ())),
        )
        if blocking is not None:
            reason, detail = blocking
            decisions.append(ShiftDecision(
                key=job.key, cohort=job.cohort, action='FREEZE', reason=reason,
                detail=detail, original_estimate=job.estimate, revised_estimate=None,
                cost=job.cost, evidence=evidence))
            continue
        delta = deltas.get(job.cohort)
        if delta is None or max(abs(part) for part in delta) > DRIFT_DELTA_LIMIT:
            decisions.append(ShiftDecision(
                key=job.key, cohort=job.cohort, action='FREEZE',
                reason='DRIFT_NOT_CORRECTABLE',
                detail='偏移位移超出位置校正可吸收范围，冻结待现场复核',
                original_estimate=job.estimate, revised_estimate=None,
                cost=job.cost, evidence=evidence))
            continue
        revised, applied = recalculate(vector, delta, kernel, feature_stamp, verification)
        decisions.append(ShiftDecision(
            key=job.key, cohort=job.cohort, action='RECALCULATE',
            reason='DRIFT_RECENTERED',
            detail='冻结模型位置校正后重算，原评分作废，按重算分继续流转',
            original_estimate=job.estimate, revised_estimate=revised,
            cost=job.cost, evidence=(*evidence, ('校正位移', applied))))

    in_scope = sum(1 for job in dispatched if scope.touches(job.key, job.cohort))
    return ShiftControlReport(feature_stamp, scope, studies, tuple(decisions), in_scope)


def apply_shift_control(plan, report):
    """把冻结/放行结论落到派单方案；冻结单保留工位占用，放行单更新评分。"""
    frozen = set(report.frozen_keys)
    revised = {item.key: item for item in report.decisions if item.action == 'RECALCULATE'}
    jobs = []
    for job in plan.jobs:
        if job.key in frozen:
            decision = next(item for item in report.decisions if item.key == job.key)
            jobs.append(job.change('冻结', decision.reason, decision.detail))
        elif job.key in revised:
            decision = revised[job.key]
            jobs.append(replace(
                job, estimate=decision.revised_estimate, reason='DRIFT_RECENTERED',
                summary=f"重算放行（原评分 {job.estimate:.4f} 已作废）"))
        else:
            jobs.append(job)
    return replace(plan, jobs=tuple(jobs))


def control_evidence(report):
    """工作台环节凭证：偏移边界、冻结清单、放行清单。"""
    rows = [
        {'作业区': item.cohort, '当班单量': item.live_samples, '训练单量': item.train_samples,
         '校准实绩': item.calibration_samples, '最大分布差': item.max_gap,
         '主导维度': item.driver, '参考状态': '未登记' if item.novel else ('偏移' if item.shifted else '稳定')}
        for item in report.cohorts]
    rows.append({'作业区': '—', '当班单量': '—', '训练单量': '—', '校准实绩': '—',
                 '最大分布差': '—', '主导维度': '—',
                 '参考状态': f"偏移边界：{len(report.scope.shifted_cohorts)} 个作业区，{len(report.scope.scoped_keys)} 张单据"})
    for decision in report.decisions:
        rows.append({
            '单号': decision.key, '作业区': decision.cohort,
            '处置': '冻结' if decision.action == 'FREEZE' else '重算放行',
            '原因码': decision.reason, '原评分': decision.original_estimate,
            '重算评分': '—' if decision.revised_estimate is None else decision.revised_estimate,
            '工位用量': decision.cost, '依据': decision.detail})
    return rows
