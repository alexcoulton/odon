use super::*;
#[test]
fn deep_link_parse_filters_and_generation_are_actor_owned() {
    let channels = spawn_test_actor();
    let url = "odon://open?roi=roi-a&visible_channels=DAPI%7CCD3&object_filters=phenotype%3Aimmune&object_filter_logic=all";
    let (parse, parse_rx) = request("deep_links.parse", json!({"url":url}));
    channels.request_tx.send(parse).unwrap();
    let parsed = parse_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(parsed["valid"], true);
    assert_eq!(parsed["request"]["roi"], "roi-a");

    let (filters, filters_rx) = request("deep_links.filters.get", json!({"url":url}));
    channels.request_tx.send(filters).unwrap();
    let filters = filters_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(filters["object_filters"][0]["property_key"], "phenotype");
    assert_eq!(filters["object_filters"][0]["query"], "immune");

    let (generate, generate_rx) = request(
        "deep_links.generate",
        json!({"request":parsed["request"].clone()}),
    );
    channels.request_tx.send(generate).unwrap();
    let generated = generate_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(generated["source"], "request");
    assert!(
        generated["url"]
            .as_str()
            .unwrap()
            .starts_with("odon://open?")
    );
    assert_eq!(channels.legacy_rx.len(), 0);
}
