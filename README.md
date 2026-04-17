# 🚀 EKF-SNPID: Control Adaptativo Bajo Demanda con Supervisión Energética

Este repositorio contiene la implementación de un controlador avanzado para robots móviles omnidireccionales (KUKA youBot). La propuesta central es un controlador **PID de Neurona Única (SNPID)** cuyas ganancias se sintonizan dinámicamente mediante un **Filtro de Kalman Extendido (EKF)**, optimizado para lidiar con saturación, obstáculos móviles y cambios súbitos de carga.

---

## 📸 Demo / Preview

<img width="1920" height="1046" alt="trayectoria" src="https://github.com/user-attachments/assets/7fe807ab-9fd2-4d7d-8760-60e8fe4ade24" />

*Comparativa de seguimiento de trayectoria con una pausa entre PID, EKF-SNPID y el controlador propuesto*

---

## 🛠️ Tecnologías usadas

- **Simuladores:** MATLAB y CoppeliaSim.
- **Lenguajes:** Python  y MATLAB.
- **Algoritmos:** Filtro de Kalman Extendido (EKF), Redes Neuronales Hamiltonianas, Lógica de Histéresis, Anti-windup.

---

## ⚙️ Innovaciones del Proyecto

El sistema no solo ajusta ganancias, sino que "decide" cuándo aprender mediante:

1. **Supervisión Energética:** Una red neuronal ligera predice el **Hamiltoniano** del sistema en línea. Si hay discrepancias con el Hamiltoniano físico, detecta un cambio de planta (como una variación de carga) y activa la readaptación.
2. **Lógica de Histéresis (Freeze/Unfreeze):** Congela el aprendizaje de ganancias cuando el error está dentro de márgenes aceptables para evitar la sobreexplotación de las ganancias.
3. **Manejo de Pausas:** Capacidad de detener el seguimiento ante obstáculos y reanudar con suavidad, evitando la acumulación del error integral.
4. **Compuesto de Estabilidad (EWMA):** Filtro para evitar reacciones prematuras ante ruidos o picos momentáneos en el error y el re-entreno de la red en línea.

---

## 📂 Estructura del Repositorio

### Control y Estimación (MATLAB)
- `EFK_SNPID_KUKA_HAMIL.m`: Script para la supervisión de energía Hamiltoniana del sistema con el EKF_SNPID.
- `EFK_SNPID_KUKA_PAUSA.m`: Implementación específica del EFK_SNPID para escenarios con obstáculos móviles que requieren pausas en el seguimiento.
- `PID_KUKA.m` / `PID_KUKA_PAUSA.m`: Controladores de referencia para benchmarking y pruebas de pausa.
- `comparativatrayectoria.m`: Genera las métricas de desempeño entre los tres controladores (RMSE, IAE/ISE, picos por eje y porcentaje de tolerancia).
- `comparativatrayectoriapausa.m`: Genera las métricas de desempeño entre los tres controladores ante una pausa debida a un obstáculo móvil (RMSE, IAE/ISE, picos por eje y porcentaje de tolerancia).

### Inteligencia Artificial (Python)
- `RedHamil.py`: Definición de la arquitectura de la red neuronal que predice el Hamiltoniano a partir de $(u_x, u_y, u_\theta)$.
- `PruebaRedHamil.py`: Script de entrenamiento y validación de la red con datos post-evento.
- `DatasetHamil.py`: Script para general el dataset para la red.
- `prueba_masa.py`: Script del controlador propuesto en dónde cambia la masa del sistema.
- `prueba_pausa.py`: Script en dónde se genera una interrupción o pausa al robot mediante un obstáculo móvil"".
- `prueba_red.py` : Script del controlador propuesto para validar predicción de la red
---

## ⚙️ Instalación y Uso

### Requisitos
- MATLAB R2021b o superior (Control System Toolbox).
- Python 3.8+ (PyTorch/TensorFlow, NumPy, Matplotlib, Pandas).
- CoppeliaSim .
  
### Pasos
1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Fidedpro888/PID-Adaptativo-Bajo-Demanda-Mediante-Redes-Neuronales.git](https://github.com/Fidedpro888/PID-Adaptativo-Bajo-Demanda-Mediante-Redes-Neuronales.git)

2. **Instalar las librerias previamente mencionadas**

3. **Abrir el entorno de simulación de Coppelia en caso de usar un .py**
   - `prueba_masa.ttt` : Entorno para el controlador con cambios de masa, pausa "imaginaria", poner a prueba la red, etc.
   - `Pruebacubo.ttt` : Entorno para el controlador con una pausa física mediante un obstáculo móvil.

   Iniciar coppelia y después el código de python, sin olvidar de correr primero `DatasetHamil.py` y luego `RedHamil.py`

4. **Para códgio de Matlab solamente es correrlo**

