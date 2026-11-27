from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CohortShift:
    """单个作业区相对训练参考的偏移证据。"""
    cohort: str
    live_samples: int
    train_samples: int
    calibration_samples: int
    max_gap: float
    driver: str
    shifted: bool
    novel: bool

    def __post_init__(self):
        if self.live_samples <= 0:
            raise ValueError('作业区缺少当班样本')
        if not 0 <= self.max_gap <= 1:
            raise ValueError('分区累积分布差须落在 0 至 1')


@dataclass(frozen=True, slots=True)
class ShiftScope:
    """偏移边界：命中的作业区与单据，以及命中依据。"""
    shifted_cohorts: tuple[str, ...]
    scoped_keys: tuple[str, ...]
    unit_reasons: dict

    def touches(self, key, cohort):
        return cohort in self.shifted_cohorts or key in self.unit_reasons


@dataclass(frozen=True, slots=True)
class ShiftDecision:
    """一张已派出单据在偏移范围内的冻结或放行结论。"""
    key: str
    cohort: str
    action: str            # FREEZE 或 RECALCULATE
    reason: str
    detail: str
    original_estimate: float
    revised_estimate: float | None
    cost: int
    evidence: tuple[tuple[str, object], ...]

    def __post_init__(self):
        if self.action not in ('FREEZE', 'RECALCULATE'):
            raise ValueError('偏移处置只能是冻结或重算放行')
        if self.action == 'RECALCULATE' and self.revised_estimate is None:
            raise ValueError('放行单必须给出重算估计')
        if self.cost < 1:
            raise ValueError('工位用量必须为正')


@dataclass(frozen=True, slots=True)
class ShiftControlReport:
    signature: str
    scope: ShiftScope
    cohorts: tuple[CohortShift, ...]
    decisions: tuple[ShiftDecision, ...]
    dispatched_in_scope: int

    def __post_init__(self):
        if not self.signature:
            raise ValueError('缺少特征版本凭证')
        actions = tuple(item.action for item in self.decisions)
        if len(actions) != self.dispatched_in_scope:
            raise ValueError('偏移范围内每张已派出单都须有结论')
        if any(action not in ('FREEZE', 'RECALCULATE') for action in actions):
            raise ValueError('存在未登记的偏移处置')

    @property
    def frozen_keys(self):
        return tuple(item.key for item in self.decisions if item.action == 'FREEZE')

    @property
    def released_keys(self):
        return tuple(item.key for item in self.decisions if item.action == 'RECALCULATE')
