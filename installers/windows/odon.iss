#ifndef SourceDir
#define SourceDir "."
#endif

#ifndef OutputDir
#define OutputDir "."
#endif

#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E9EE575-4C9D-4E9E-9E5E-5E82D0966B45}
AppName=Odon
AppVersion={#AppVersion}
AppPublisher=Odon Developers
AppPublisherURL=https://github.com/alexcoulton/odon
AppSupportURL=https://github.com/alexcoulton/odon/issues
AppUpdatesURL=https://github.com/alexcoulton/odon/releases
DefaultDirName={autopf}\Odon
DefaultGroupName=Odon
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=OdonSetup-{#AppVersion}-windows-x86_64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern
SetupIconFile={#SourceDir}\assets\odon.ico
UninstallDisplayIcon={app}\assets\odon.ico

[Files]
Source: "{#SourceDir}\odon.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\odon_mcp.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\examples\*"; DestDir: "{app}\examples"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Odon"; Filename: "{app}\odon.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\odon.ico"
Name: "{autodesktop}\Odon"; Filename: "{app}\odon.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\odon.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Registry]
Root: HKA; Subkey: "Software\Classes\odon"; ValueType: string; ValueName: ""; ValueData: "URL:Odon Protocol"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\odon"; ValueType: string; ValueName: "URL Protocol"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\odon\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\odon.ico"
Root: HKA; Subkey: "Software\Classes\odon\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\odon.exe"" ""%1"""

[Run]
Filename: "{app}\odon.exe"; Description: "Launch Odon"; Flags: nowait postinstall skipifsilent
