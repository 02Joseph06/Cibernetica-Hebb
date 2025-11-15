# 1. Definición del Problema
✨ **Clasificación de Mobs de Minecraft usando Aprendizaje Hebbiano** ✨

---

## Participantes
- Juan Holguín  
- Juan Vásquez  
- Luigi Rincón  
- Samuel Moya  
- Sara León  
- Uriel Rodríguez  

---


## ⭐ 1. Definición del Problema
Se plantea una tarea sencilla de clasificación utilizando aprendizaje hebbiano: clasificar imágenes de tres mobs de Minecraft (Aldeano, Aldeano Zombi y Pillager).

Las imágenes se procesan convirtiéndolas a escala de grises, reduciéndolas a 8×8 píxeles y normalizándolas, produciendo un vector de 64 entradas por imagen.


**Clasificar imágenes reales de mobs de Minecraft** en tres categorías:

- 🟦 **Aldeano**  
- 🟩 **Aldeano Zombi**  
- 🟥 **Pillager**

---


---

## ⭐ 2. Diseño de la Red
La red se estructura como un modelo hebbiano multicapa:

- 64 neuronas de entrada (píxeles de la imagen)
- 8 neuronas ocultas con activación *tanh*
- 3 neuronas de salida (una por clase)

El aprendizaje sigue la regla de Hebb, reforzando conexiones por coactivación: *"las neuronas que se activan juntas se conectan juntas"*.

---

## ⭐ 3. Implementación del Modelo
El modelo se implementa en Python y permite:

- Inicializar pesos aleatorios para **W1** y **W2**.
- Entrenar mediante aprendizaje hebbiano, reforzando las conexiones con cada muestra.
- Probar el modelo: una nueva imagen atraviesa la red y la salida con mayor activación determina la clase predicha.
- Visualizar la evolución de los pesos usando *Matplotlib*.

---

## ⭐ 4. Evaluación y Análisis de Resultados
La red obtuvo los siguientes resultados:

- Clasificación correcta para **Aldeano** y **Aldeano Zombi**.
- Confusión del **Pillager** con **Aldeano Zombi**.
- Precisión final aproximada: **66%**.

Conclusiones:
- La red formó prototipos básicos.
- Limitaciones debidas a pocas imágenes por clase, baja resolución y la ausencia de corrección de errores en la regla de Hebb.

Posibles mejoras:
- Aumentar la resolución.
- Incorporar más ejemplos por clase.
- Usar características visuales más ricas o filtros previos.
