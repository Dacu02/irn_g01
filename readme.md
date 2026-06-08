# irn_g01 - ArUco in Gazebo Ignition

Questo pacchetto include un esempio minimo funzionante per mostrare un marker ArUco in Gazebo Ignition/Gazebo Sim.

## File aggiunti
- `worlds/aruco_demo.sdf`: mondo di esempio.
- `models/aruco_marker/model.sdf`: modello statico con texture del marker.
- `models/aruco_marker/materials/textures/aruco_4x4_00.png`: immagine usata come texture.

## Avvio
```bash
source install/setup.bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(ros2 pkg prefix irn_g01)/share/irn_g01
gz sim $(ros2 pkg prefix irn_g01)/share/irn_g01/worlds/aruco_demo.sdf
```

## Se vuoi usare il tuo ArUco (PDF/SVG/PNG)
Ignition applica direttamente texture raster. Quindi:
1. se parti da PDF/SVG, converti in PNG;
2. sostituisci `models/aruco_marker/materials/textures/aruco_4x4_00.png`;
3. mantieni lo stesso path oppure aggiorna `albedo_map` in `model.sdf`.
