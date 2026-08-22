use super::*;
#[test]
fn multi_viewport_active_key_union_deduplicates_and_retains_peer_work() {
    let mut union = HashSet::new();
    merge_viewport_active_keys(&mut union, [1u8, 2, 3]);
    merge_viewport_active_keys(&mut union, [3u8, 4, 5]);
    assert_eq!(union, HashSet::from([1u8, 2, 3, 4, 5]));
}
