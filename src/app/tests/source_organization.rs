use std::fs;
use std::path::{Path, PathBuf};

fn source(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path.as_ref())
        .unwrap_or_else(|error| panic!("read {}: {error}", path.as_ref().display()))
}

fn rust_files(path: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn app_source_stays_split_by_responsibility() {
    let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app");
    let facade = source(app_dir.join("mod.rs"));
    assert!(
        facade.lines().count() <= 4_000,
        "src/app/mod.rs has regrown into an implementation monolith"
    );
    assert!(
        !facade.contains("impl eframe::App for OmeZarrViewerApp"),
        "the frame lifecycle belongs in app/update.rs"
    );

    for path in rust_files(&app_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 3_500,
            "{} is too large; split it at a responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "production app tests belong under app/tests: {}",
            path.display()
        );
    }
}

#[test]
fn legacy_control_is_an_explicit_migration_boundary() {
    let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app");
    let legacy = source(app_dir.join("legacy_control/mod.rs"));
    assert!(legacy.contains("Temporary compatibility boundary"));

    let actor_facade =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/control/actor/mod.rs"));
    assert!(
        actor_facade.lines().count() <= 300,
        "the control actor façade must not regain domain handlers"
    );
}
