fn main() {
    #[cfg(target_os = "windows")]
    {
        winresource::WindowsResource::new()
            .set_icon("assets/odon.ico")
            .set("FileDescription", "Odon")
            .set("ProductName", "Odon")
            .compile()
            .expect("failed to embed Windows icon resource");
    }
}
