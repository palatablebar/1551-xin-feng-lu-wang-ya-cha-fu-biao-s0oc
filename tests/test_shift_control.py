from dataclasses import replace
from types import SimpleNamespace

import pytest

from ductwatch.building_cohorts.service import fit_baseline, scan_current
from ductwatch.fan_replay.service import prepare_verification, validate_predictions
from ductwatch.filter_inspection.service import plan_jobs
from ductwatch.flow_subspace.service import fit_model, predict
from ductwatch.pipeline import open_fixture
from ductwatch.duct_checklists.service import load_samples
from ductwatch.pressure_epochs.service import build_windows
from ductwatch.ventilation_factors.service import fit_features, transform_windows
from ductwatch.shift_control import (
    COVERAGE_FLOOR,
    DRIFT_DELTA_LIMIT,
    evaluate_shift_control,
    apply_shift_control,
    control_evidence,
)
from ductwatch.shift_control.contracts import ShiftControlReport
from ductwatch.shift_control import service as shift_service


def _pipeline_state():
    sample = open_fixture()
    intake = load_samples(sample['records'], sample['as_of'])
    windows = build_windows(intake.entries)
    state = fit_features(windows.train)
    features = transform_windows(windows, state)
    kernel = fit_model(features.train, state.fingerprint)
    calibration = prepare_verification(
        predict(kernel, features.calibration, state.fingerprint))
    baseline = fit_baseline(features.train, state.fingerprint)
    return SimpleNamespace(
        sample=sample, state=state, features=features, kernel=kernel,
        calibration=calibration, baseline=baseline,
        calibration_vectors=features.calibration,
        train_vectors=features.train, live_vectors=features.live)


@pytest.fixture(scope='session')
def bench():
    return _pipeline_state()


def _plan(ctx, live_override, *, capacity=60):
    drift = scan_current(ctx.baseline, live_override, ctx.state.fingerprint)
    checked = validate_predictions(
        predict(ctx.kernel, live_override, ctx.state.fingerprint), ctx.calibration)
    return plan_jobs(checked, drift.domain, capacity, 'focus')


def _report(ctx, live_override, plan):
    return evaluate_shift_control(
        ctx.baseline, live_override, ctx.calibration_vectors, ctx.kernel,
        ctx.state.fingerprint, ctx.calibration, ctx.train_vectors, live_override, plan)


def _shift_cohort(live, cohort, bump):
    return tuple(
        replace(v, values=tuple(min(12.0, max(-12.0, part + bump)) for part in v.values))
        if v.cohort == cohort else v for v in live)


def _dispatched_cohort(plan):
    return next(job.cohort for job in plan.jobs if job.state == '已分派')


def test_stable_preset_has_no_scope_decisions(bench):
    plan = _plan(bench, bench.live_vectors)
    report = _report(bench, bench.live_vectors, plan)
    assert report.frozen_keys == () and report.released_keys == ()
    assert report.dispatched_in_scope == 0
    # 未知区虽在边界内，但其中没有已派出单。
    assert report.scope.shifted_cohorts == ('临时接管区',)


def test_unknown_cohort_dispatched_ticket_is_frozen(bench):
    live = bench.live_vectors
    cohort = _dispatched_cohort(_plan(bench, live))
    job_key = next(j.key for j in _plan(bench, live).jobs
                   if j.state == '已分派' and j.cohort == cohort)
    moved = tuple(replace(v, cohort='未登记机房') if v.key == job_key else v for v in live)
    plan = _plan(bench, moved)
    # 未知区单在派单环节转人工，强制回到已派出以检验冻结判定本身。
    plan = replace(plan, jobs=tuple(
        replace(j, state='已分派') if j.key == job_key else j for j in plan.jobs))
    report = _report(bench, moved, plan)
    decision = next(item for item in report.decisions if item.key == job_key)
    assert decision.action == 'FREEZE' and decision.reason == 'NO_COHORT_REFERENCE'
    assert '未登记机房' in report.scope.shifted_cohorts


def test_double_missing_dispatched_ticket_is_frozen(bench):
    live = bench.live_vectors
    plan = _plan(bench, live)
    target = next(j.key for j in plan.jobs if j.state == '已分派')
    broken = tuple(replace(v, missing=('pressure_pa', 'flow_cmh')) if v.key == target else v
                   for v in live)
    plan = replace(plan, jobs=tuple(
        replace(j, state='已分派') if j.key == target else j for j in plan.jobs))
    report = _report(bench, broken, plan)
    decision = next(item for item in report.decisions if item.key == target)
    assert decision.action == 'FREEZE' and decision.reason == 'MEASUREMENT_GAP'


def test_double_envelope_breach_freezes_dispatched_ticket(bench):
    live = bench.live_vectors
    plan = _plan(bench, live)
    target = next(j.key for j in plan.jobs if j.state == '已分派')
    breached = tuple(
        replace(v, excursions=('flow_resistance', 'fan_delivery')) if v.key == target else v
        for v in live)
    plan = replace(plan, jobs=tuple(
        replace(j, state='已分派') if j.key == target else j for j in plan.jobs))
    report = _report(bench, breached, plan)
    decision = next(item for item in report.decisions if item.key == target)
    assert decision.action == 'FREEZE' and decision.reason == 'ENVELOPE_BREACH'


def test_whole_cohort_shift_recalculates_covered_tickets(bench):
    plan = _plan(bench, bench.live_vectors)
    cohort = _dispatched_cohort(plan)
    shifted = _shift_cohort(bench.live_vectors, cohort, 0.9)
    report = _report(bench, shifted, _plan(bench, shifted))
    assert cohort in report.scope.shifted_cohorts
    in_scope = [d for d in report.decisions if d.cohort == cohort]
    assert in_scope and all(d.action == 'RECALCULATE' for d in in_scope)
    assert all(d.revised_estimate is not None for d in in_scope)


def test_uncorrectable_shift_freezes(bench):
    plan = _plan(bench, bench.live_vectors)
    cohort = _dispatched_cohort(plan)
    shifted = _shift_cohort(bench.live_vectors, cohort, DRIFT_DELTA_LIMIT + 0.5)
    report = _report(bench, shifted, _plan(bench, shifted))
    decisions = [d for d in report.decisions if d.cohort == cohort]
    assert decisions
    assert all(d.action == 'FREEZE' and d.reason == 'DRIFT_NOT_CORRECTABLE'
               for d in decisions)


def test_stale_signature_is_rejected(bench):
    plan = _plan(bench, bench.live_vectors)
    with pytest.raises(ValueError, match='基准'):
        evaluate_shift_control(
            bench.baseline, bench.live_vectors, bench.calibration_vectors, bench.kernel,
            'stale', bench.calibration, bench.train_vectors, bench.live_vectors, plan)


def test_low_calibration_coverage_forces_freeze(bench, monkeypatch):
    plan = _plan(bench, bench.live_vectors)
    cohort = _dispatched_cohort(plan)
    shifted = _shift_cohort(bench.live_vectors, cohort, 0.9)
    original = shift_service.cohort_drift

    def thin_coverage(reference, observed, calibration):
        studies = original(reference, observed, calibration)
        return tuple(
            replace(item, calibration_samples=COVERAGE_FLOOR - 1, shifted=True)
            if item.cohort == cohort else item for item in studies)

    monkeypatch.setattr(shift_service, 'cohort_drift', thin_coverage)
    report = _report(bench, shifted, _plan(bench, shifted))
    decisions = [d for d in report.decisions if d.cohort == cohort]
    assert decisions
    assert all(d.action == 'FREEZE' and d.reason == 'COVERAGE_INSUFFICIENT'
               for d in decisions)


def test_freeze_reason_rules_are_explicit(bench):
    from ductwatch.shift_control.contracts import CohortShift as Study
    live = bench.live_vectors
    vector = next(v for v in live if not v.missing and not v.excursions)
    covered = Study(vector.cohort, 5, 5, COVERAGE_FLOOR, 0.7, 'x', True, False)
    assert shift_service.freeze_reason(vector, covered) is None
    thin = replace(covered, calibration_samples=COVERAGE_FLOOR - 1)
    assert shift_service.freeze_reason(vector, thin)[0] == 'COVERAGE_INSUFFICIENT'
    novel = replace(covered, novel=True, calibration_samples=0)
    assert shift_service.freeze_reason(vector, novel)[0] == 'NO_COHORT_REFERENCE'
    gappy = replace(vector, missing=('pressure_pa', 'flow_cmh'))
    assert shift_service.freeze_reason(gappy, covered)[0] == 'MEASUREMENT_GAP'
    breached = replace(vector, excursions=('flow_resistance', 'fan_delivery'))
    assert shift_service.freeze_reason(breached, covered)[0] == 'ENVELOPE_BREACH'


def test_apply_control_keeps_capacity_and_updates_released_score(bench):
    plan = _plan(bench, bench.live_vectors)
    cohort = _dispatched_cohort(plan)
    shifted = _shift_cohort(bench.live_vectors, cohort, 0.9)
    shifted_plan = _plan(bench, shifted)
    report = _report(bench, shifted, shifted_plan)
    controlled = apply_shift_control(shifted_plan, report)
    assert controlled.balance == shifted_plan.balance
    released = next(j for j in controlled.jobs if j.key in report.released_keys)
    original = next(j for j in shifted_plan.jobs if j.key == released.key)
    assert released.state == '已分派' and released.reason == 'DRIFT_RECENTERED'
    assert released.estimate == report.decisions[0].revised_estimate
    assert released.estimate != original.estimate or released.reason != original.reason


def test_same_inputs_produce_identical_lists(bench):
    plan = _plan(bench, bench.live_vectors)
    cohort = _dispatched_cohort(plan)
    shifted = _shift_cohort(bench.live_vectors, cohort, 0.9)
    first = _report(bench, shifted, _plan(bench, shifted))
    second = _report(bench, shifted, _plan(bench, shifted))
    assert first == second
    assert (first.frozen_keys, first.released_keys) == (second.frozen_keys, second.released_keys)
    rows = control_evidence(first)
    assert any(row.get('处置') in ('冻结', '重算放行') for row in rows)
    assert isinstance(first, ShiftControlReport)
