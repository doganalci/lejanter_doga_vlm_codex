# Labeling Guide

## Facade Regions

Start with one class:

```text
facade_region
```

Label each visually distinct facade plane as a separate instance. If a building shows front and side faces, draw two polygons. If multiple buildings appear in one image, draw one polygon per visible facade plane.

Do not include sky, ground, trees, cars, or unrelated surroundings. Include facade surface behind windows/panels when it belongs to the architectural plane.

## Elements

Element classes should describe the architectural legend item, not only the geometry. Prefer stable names such as:

```text
window_type_a
window_type_b
panel_type_a
door
balcony_railing
mechanical_louver
stone_cladding
glass_curtain_wall
```

If two visual items will be counted differently in reports, they should be different classes.

## Quality Rules

- Keep class names stable after training starts.
- Use polygons for facade regions.
- Use boxes for simple rectangular items, polygons for irregular facade materials.
- Mark partially visible repeated items if they should be counted in the final report.
- Keep a small validation set with representative building types.
