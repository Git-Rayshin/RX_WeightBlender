# RX Weight Blender

RX Weight Blender is a tool for quickly blending skin weights with selected vertices in Maya.

This public version is simple and fast tool utilizes the Maya API.

![maya2020](https://img.shields.io/badge/Maya_2020-tested-brightgreen.svg)
![maya2022](https://img.shields.io/badge/Maya_2022-tested-brightgreen.svg)
![maya2023](https://img.shields.io/badge/Maya_2023-tested-brightgreen.svg)
![maya2024](https://img.shields.io/badge/Maya_2024-tested-brightgreen.svg)

![Windows](https://img.shields.io/badge/Windows-tested-blue)

<div style="text-align: center;">
    <img src="https://github.com/Git-Rayshin/RX_WeightBlender/assets/115437984/5f27dfd9-7dbb-4dce-95a0-9b707446304d" alt="demo" height="500">
</div>

-------------------

## Features

- Quickly match influences with selected vertices.
- One-click selected vertex weight blend to a 50% 50% weight of the last two selected vertices
- Two types of sliders for rapid weight editing.
- Right-click on the slider to set a specific value.

## Installation

Place the `rx_weight_blender` directory in one of Maya's Python script directories.

Alternatively, add the directory of your choice to the PYTHONPATH environment variable, then place
the `rx_weight_blender` directory in that directory.

## Usage

Launch the GUI by running the following:

```python
import rx_weight_blender.ui as wb

wb.show()
```

- One-way blend: Select the vertices you want to edit, then select the target vertex last, and start blending!
- Two-way blend: Select the vertices you want to edit, then select the first target, then the second target, and start
  blending!

## Revisions

### 1.0.0

Initial release of RX Weight Blender

## Author

[Ruixin He](https://github.com/Git-Rayshin)

## License

MIT License
