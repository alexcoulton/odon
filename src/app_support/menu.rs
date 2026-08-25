use anyhow::Context;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeMenuCommand {
    pub command_id: String,
    pub checked: Option<bool>,
}

#[cfg(target_os = "macos")]
pub struct NativeMenu {
    _menu: muda::Menu,
    commands: Vec<(muda::MenuId, String)>,
    checked_commands: Vec<(muda::MenuId, String, muda::CheckMenuItem)>,
}

#[cfg(not(target_os = "macos"))]
pub struct NativeMenu;

impl NativeMenu {
    #[cfg(target_os = "macos")]
    pub fn init(app_name: &str, command_surface: &Value) -> anyhow::Result<Self> {
        use std::collections::HashMap;

        use muda::Menu;

        let commands = command_surface
            .get("commands")
            .and_then(Value::as_array)
            .context("command surface has no commands")?
            .iter()
            .filter_map(|command| Some((command.get("id")?.as_str()?, command)))
            .collect::<HashMap<_, _>>();
        let menu_tree = command_surface
            .get("menu")
            .context("command surface has no platform menu")?;
        let top_level = menu_tree
            .get("children")
            .and_then(Value::as_array)
            .context("platform menu has no top-level menus")?;

        let menu = Menu::new();
        let mut commands_by_menu_id = Vec::new();
        let mut checked_commands = Vec::new();
        for node in top_level {
            let submenu = build_submenu(
                node,
                app_name,
                &commands,
                &mut commands_by_menu_id,
                &mut checked_commands,
            )?;
            menu.append(&submenu)
                .context("failed to append declarative top-level menu")?;
        }
        menu.init_for_nsapp();
        Ok(Self {
            _menu: menu,
            commands: commands_by_menu_id,
            checked_commands,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn init(_app_name: &str, _command_surface: &Value) -> anyhow::Result<Self> {
        Ok(Self)
    }

    #[cfg(target_os = "macos")]
    pub fn drain_commands(&self) -> Vec<NativeMenuCommand> {
        let mut out = Vec::new();
        while let Ok(event) = muda::MenuEvent::receiver().try_recv() {
            let id = event.id();
            if let Some((_, command_id, item)) = self
                .checked_commands
                .iter()
                .find(|(candidate, _, _)| candidate == id)
            {
                out.push(NativeMenuCommand {
                    command_id: command_id.clone(),
                    checked: Some(item.is_checked()),
                });
            } else if let Some((_, command_id)) =
                self.commands.iter().find(|(candidate, _)| candidate == id)
            {
                out.push(NativeMenuCommand {
                    command_id: command_id.clone(),
                    checked: None,
                });
            }
        }
        out
    }

    #[cfg(not(target_os = "macos"))]
    pub fn drain_commands(&self) -> Vec<NativeMenuCommand> {
        Vec::new()
    }
}

#[cfg(target_os = "macos")]
fn build_submenu(
    node: &Value,
    app_name: &str,
    commands: &std::collections::HashMap<&str, &Value>,
    commands_by_menu_id: &mut Vec<(muda::MenuId, String)>,
    checked_commands: &mut Vec<(muda::MenuId, String, muda::CheckMenuItem)>,
) -> anyhow::Result<muda::Submenu> {
    use muda::{CheckMenuItem, MenuItem, PredefinedMenuItem, Submenu};

    let title = node
        .get("title")
        .and_then(Value::as_str)
        .context("menu title is missing")?;
    let title = if node.get("id").and_then(Value::as_str) == Some("menu:application") {
        app_name
    } else {
        title
    };
    let submenu = Submenu::new(title, true);
    let children = node
        .get("children")
        .and_then(Value::as_array)
        .context("menu children are missing")?;
    for child in children {
        match child.get("type").and_then(Value::as_str) {
            Some("menu") => {
                let nested = build_submenu(
                    child,
                    app_name,
                    commands,
                    commands_by_menu_id,
                    checked_commands,
                )?;
                submenu
                    .append(&nested)
                    .context("failed to append nested menu")?;
            }
            Some("separator") => {
                submenu
                    .append(&PredefinedMenuItem::separator())
                    .context("failed to append menu separator")?;
            }
            Some("command") => {
                let command_id = child
                    .get("command_id")
                    .and_then(Value::as_str)
                    .context("menu command ID is missing")?;
                let command = commands
                    .get(command_id)
                    .copied()
                    .with_context(|| format!("menu references unknown command '{command_id}'"))?;
                if command.pointer("/state/visible").and_then(Value::as_bool) == Some(false) {
                    continue;
                }
                if command_id == "app.about" {
                    submenu
                        .append(&PredefinedMenuItem::about(
                            Some(&format!("About {app_name}")),
                            None,
                        ))
                        .context("failed to append About command")?;
                    continue;
                }
                let title = child
                    .get("label")
                    .and_then(Value::as_str)
                    .or_else(|| command.get("title").and_then(Value::as_str))
                    .context("command title is missing")?;
                let accelerator = command_accelerator(command)?;
                let enabled = command
                    .pointer("/state/enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                if let Some(checked) = command.pointer("/state/checked").and_then(Value::as_bool) {
                    let item = CheckMenuItem::new(title, enabled, checked, accelerator);
                    checked_commands.push((
                        item.id().clone(),
                        command_id.to_string(),
                        item.clone(),
                    ));
                    submenu
                        .append(&item)
                        .context("failed to append checked command")?;
                } else {
                    let item = MenuItem::new(title, enabled, accelerator);
                    commands_by_menu_id.push((item.id().clone(), command_id.to_string()));
                    submenu
                        .append(&item)
                        .context("failed to append menu command")?;
                }
            }
            kind => anyhow::bail!("unsupported declarative menu node type {kind:?}"),
        }
    }
    Ok(submenu)
}

#[cfg(target_os = "macos")]
fn command_accelerator(command: &Value) -> anyhow::Result<Option<muda::accelerator::Accelerator>> {
    use muda::accelerator::{Accelerator, Code, Modifiers};

    let Some(shortcut) = command.get("shortcut").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let key = shortcut
        .get("key")
        .and_then(Value::as_str)
        .context("shortcut key is missing")?;
    let code = match key.to_ascii_lowercase().as_str() {
        "a" => Code::KeyA,
        "b" => Code::KeyB,
        "c" => Code::KeyC,
        "d" => Code::KeyD,
        "e" => Code::KeyE,
        "f" => Code::KeyF,
        "g" => Code::KeyG,
        "h" => Code::KeyH,
        "i" => Code::KeyI,
        "j" => Code::KeyJ,
        "k" => Code::KeyK,
        "l" => Code::KeyL,
        "m" => Code::KeyM,
        "n" => Code::KeyN,
        "o" => Code::KeyO,
        "p" => Code::KeyP,
        "q" => Code::KeyQ,
        "r" => Code::KeyR,
        "s" => Code::KeyS,
        "t" => Code::KeyT,
        "u" => Code::KeyU,
        "v" => Code::KeyV,
        "w" => Code::KeyW,
        "x" => Code::KeyX,
        "y" => Code::KeyY,
        "z" => Code::KeyZ,
        "comma" => Code::Comma,
        _ => anyhow::bail!("unsupported platform shortcut key '{key}'"),
    };
    let mut modifiers = Modifiers::empty();
    for modifier in shortcut
        .get("modifiers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        modifiers |= match modifier {
            "primary" | "super" => Modifiers::SUPER,
            "shift" => Modifiers::SHIFT,
            "alt" => Modifiers::ALT,
            "control" => Modifiers::CONTROL,
            _ => anyhow::bail!("unsupported platform shortcut modifier '{modifier}'"),
        };
    }
    Ok(Some(Accelerator::new(
        (!modifiers.is_empty()).then_some(modifiers),
        code,
    )))
}
