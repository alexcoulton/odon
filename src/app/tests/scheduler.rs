use super::*;
#[test]
fn multi_viewport_scheduler_prioritizes_active_without_starving_peer() {
    let (active_raw, active_cpu) = viewport_image_request_budgets(true, true);
    let (peer_raw, peer_cpu) = viewport_image_request_budgets(true, false);
    assert!(active_raw > peer_raw && peer_raw > 0);
    assert!(active_cpu > peer_cpu && peer_cpu > 0);
    assert_eq!(viewport_image_request_budgets(false, false), (256, 64));
}
