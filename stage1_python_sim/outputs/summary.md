# Stage 1 benchmark (20 seeds per cell, pedestrians ignore the robot)

| Scenario | Controller | Success % | Collision % | Timeout % | Time (s)* | Min dist (m) | Exposure % |
|---|---|---:|---:|---:|---:|---:|---:|
| chokepoint | goal_seeker | 0 | 100 | 0 | – | 0.58 | 18.1 |
| chokepoint | social_force | 100 | 0 | 0 | 27.7 | 1.24 | 0.3 |
| chokepoint | predictive_sf | 85 | 15 | 0 | 26.6 | 1.12 | 1.1 |
| chokepoint | learned | 95 | 0 | 5 | 23.3 | 2.12 | 0.0 |
| corridor | goal_seeker | 0 | 100 | 0 | – | 0.57 | 13.3 |
| corridor | social_force | 100 | 0 | 0 | 13.3 | 0.92 | 2.7 |
| corridor | predictive_sf | 100 | 0 | 0 | 12.2 | 1.13 | 0.0 |
| corridor | learned | 100 | 0 | 0 | 11.7 | 1.01 | 1.4 |
| crossing | goal_seeker | 10 | 90 | 0 | 11.3 | 0.59 | 11.6 |
| crossing | social_force | 55 | 45 | 0 | 12.7 | 0.82 | 7.7 |
| crossing | predictive_sf | 60 | 40 | 0 | 12.6 | 0.83 | 5.1 |
| crossing | learned | 100 | 0 | 0 | 20.0 | 1.25 | 0.2 |
| dense_social | goal_seeker | 0 | 100 | 0 | – | 0.58 | 15.0 |
| dense_social | social_force | 80 | 20 | 0 | 17.8 | 0.98 | 3.2 |
| dense_social | predictive_sf | 75 | 25 | 0 | 17.0 | 1.01 | 5.5 |
| dense_social | learned | 100 | 0 | 0 | 13.5 | 1.72 | 0.0 |

*Time to goal over successful episodes only. Distances are centre-to-centre; exposure = share of time with a pedestrian closer than 1.0 m.
