fn main() {
    // exe icon for Explorer, the taskbar and Start Menu shortcuts
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winresource::WindowsResource::new().set_icon("assets/icon.ico").compile().expect("embedding the icon");
    }
}
