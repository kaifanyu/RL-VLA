import pytest
from rl_vla.diagnostics.review_transitions import transitions


def rows(tmp_path):
    report=tmp_path/"report.json";report.write_text("{}")
    review=tmp_path/"review.json";review.write_text("{}")
    result=[]
    for policy in ("pi05_libero","rldt_50"):
        for state in (150,151):
            result.append({"task":"object_0","split":"test","policy":policy,"initial_state_id":state,
                           "seed":100000+state-150,"success":False,"reviewed":True,"failure_type":"wrong object",
                           "report":str(report),"review_source":str(review),"bddl_sha256":"same"})
    return result


def test_wrong_to_placement_is_not_a_success_recovery(tmp_path):
    data=rows(tmp_path);data[2]["failure_type"]="correct grasp but placement failure";data[3]["success"]=True
    cases,counts,summary=transitions(data,expected_count=2)
    cell=next(row for row in summary if row.get("status")=="complete_review")
    assert cell["wrong_failure_to_success"]==1
    assert cell["wrong_failure_to_placement_failure"]==1
    assert cell["recovered"]==1
    assert len(cases)==2 and sum(row["count"] for row in counts)==2


def test_unreviewed_and_ambiguous_failures_stay_separate(tmp_path):
    data=rows(tmp_path);data[2]["reviewed"]=False;data[2]["review_source"]=None
    data[3]["failure_type"]="other/ambiguous"
    cases,counts,summary=transitions(data,expected_count=2)
    assert {row["after_category"] for row in cases}=={"not reviewed","other/ambiguous"}
    assert next(row for row in summary if row.get("paired_episodes"))["status"]=="pending_review"


def test_mismatched_paired_identity_rejected(tmp_path):
    data=rows(tmp_path);data[2]["seed"]+=99
    with pytest.raises(ValueError,match="identical state/seed"):
        transitions(data,expected_count=2)
