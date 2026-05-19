# 便携版打包与迁移（Linux）

## 1) 一键打包
在项目根目录执行：

```bash
chmod +x build_portable.sh
./build_portable.sh
```

打包产物输出到：

- `dist/cognitive_training_game/`

## 2) 迁移到另一台设备
将整个目录 `dist/cognitive_training_game/` 复制到目标 Linux 设备（U 盘/SCP 均可）。

## 3) 目标设备运行
进入目录后执行：

```bash
chmod +x cognitive_training_game
./cognitive_training_game
```

## 4) 注意事项
- 目标设备无需安装 Python。
- 目标设备需要图形桌面和可用摄像头。
- 目标设备建议与打包设备保持同架构（例如都为 ARM64 或都为 x86_64）。
- 目标设备需具备常见系统动态库（如 `libGL`、`libX11`）；通常桌面 Linux 默认已有。
- 程序运行后会在程序目录自动创建 `training_data` 并保存数据。
