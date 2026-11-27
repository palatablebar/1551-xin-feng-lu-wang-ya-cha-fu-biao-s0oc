import pytest
from ductwatch.entry import app
from ductwatch.pipeline import execute, open_fixture, preview

ROUTE = '/api/ductwatch'


def test_preset_workbench_has_real_business_rows():
    output = app.test_client().get(ROUTE+'/overview').get_json()
    assert len(output['metrics']) == 6 and len(output['sections']) == 9
    assert len(output['rows']) >= 10
    assert output['model'] == 'PCA 压差残差评分'


def test_scenario_reallocates_business_capacity():
    test_client = app.test_client()
    tight = test_client.post(ROUTE+'/run',json={'capacity':1}).get_json()
    wide = test_client.post(ROUTE+'/run',json={'capacity':60,'strategy':'focus'}).get_json()
    assert tight['feature_stamp'] == wide['feature_stamp']
    assert tight['rows'] != wide['rows']
    assert tight['run_id'] != wide['run_id']


def test_windows_refuse_bad_document():
    response = app.test_client().post(ROUTE+'/run',json={'records':[{}]})
    assert response.status_code == 422 and response.get_json()['code'] == 'INVALID_BATCH'


def test_exact_health_and_template_match():
    test_client = app.test_client()
    assert test_client.get('/health').get_json() == {'status':'ok','service':'ductwatch'}
    assert '新风滤网压差复标' in test_client.get('/').get_data(as_text=True)
    assert test_client.get('/static/desk.js').status_code == 200


def test_entered_data_matches_preset_when_identical():
    result = app.test_client().post(ROUTE+'/run',json=open_fixture())
    assert result.status_code == 200
    assert result.get_json()['run_id'] == preview()['run_id']


def test_input_order_cannot_change_disposition():
    dataset = open_fixture()
    forward = execute(dataset['records'])
    backward = execute(list(reversed(dataset['records'])))
    assert forward['run_id'] == backward['run_id']
    assert forward['rows'] == backward['rows']


def test_changed_profile_fetches_matching_stage():
    web = app.test_client()
    run = web.post(ROUTE+'/run',json={'capacity':6, 'strategy':'focus'}).get_json()
    found = web.get(ROUTE+'/stages/'+run['sections'][6]['key']+'?capacity=6&strategy=focus')
    assert found.status_code == 200
    assert found.get_json() == {**run['sections'][6],'run_id':run['run_id']}


@pytest.mark.parametrize('capacity',[0,True,61,None,'8',3.5])
def test_capacity_type_check(capacity):
    response = app.test_client().post(ROUTE+'/run',json={'capacity':capacity})
    assert response.status_code == 422


def _shift_cohort_records(records, cohort, field, amount, *, cap=None):
    shifted = [dict(row) for row in records]
    for row in shifted:
        if row['day'] >= 88 and row.get('cohort') == cohort and row.get(field) is not None:
            value = row[field] + amount
            row[field] = min(cap, value) if cap is not None else value
    return shifted


def test_shift_freeze_keeps_workstation_and_voucher_balanced():
    fixture = open_fixture()
    # 大偏移超出位置校正上限：范围内已派出单冻结，仍占工位并计入凭证。
    shifted = _shift_cohort_records(fixture['records'], '地下机房', 'since_days', 60, cap=185)
    shifted = _shift_cohort_records(shifted, '地下机房', 'pm_index', 60, cap=250)
    report = execute(shifted)
    control = report['shift_control']
    assert '地下机房' in control['shifted_cohorts']
    assert control['frozen_keys'] and not control['released_keys']
    frozen_rows = [row for row in report['rows'] if row['state'] == '冻结']
    assert {row['key'] for row in frozen_rows} == set(control['frozen_keys'])
    occupied = sum(row['cost'] for row in report['rows']
                   if row['state'] in ('已分派', '冻结'))
    assert occupied + report['metrics'][-1]['value'] == report['capacity']
    # 冻结单进入设备核销凭证，处置统计仍覆盖全部当班单。
    assert sum(count for _, count in report['flow_counts']) == len(report['rows'])
    voucher = report['sections'][8]['records']
    assert any(item.get('单号') in control['frozen_keys'] and item.get('处置') == '冻结'
               for item in voucher)
    # 稳定区不受影响，照常推进。
    assert any(row['state'] == '已分派' and row['zone'] != '地下机房' for row in report['rows'])


def test_moderate_shift_recalculates_and_releases_dispatched_tickets():
    fixture = open_fixture()
    shifted = _shift_cohort_records(fixture['records'], '地下机房', 'pm_index', 35, cap=250)
    report = execute(shifted)
    control = report['shift_control']
    assert control['released_keys'] and not control['frozen_keys']
    released = [row for row in report['rows'] if row['key'] in control['released_keys']]
    assert all(row['state'] == '已分派' and row['reason'] == 'DRIFT_RECENTERED' for row in released)
    # 同一批输入重算，名单与凭证完全一致。
    repeated = execute([dict(row) for row in shifted])
    assert repeated['run_id'] == report['run_id']
    assert repeated['rows'] == report['rows']
    assert repeated['shift_control'] == control


def test_shift_lists_are_order_independent():
    fixture = open_fixture()
    shifted = _shift_cohort_records(fixture['records'], '地下机房', 'since_days', 60, cap=185)
    forward = execute(shifted)
    backward = execute(list(reversed([dict(row) for row in shifted])))
    assert forward['run_id'] == backward['run_id']
    assert forward['shift_control'] == backward['shift_control']
    assert forward['rows'] == backward['rows']
