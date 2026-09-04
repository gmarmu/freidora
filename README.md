# freidora
Libreria RISC para uso de pandas de forma simplificada

    Algunas decisiones de diseño:
      * Las operaciones se hacen por omisión inplace, de forma opuesta a pandas
      * Todas las operaciones devuelven el dataset, independientemente de si son inplace o no
      * Elimina todos los campos y índices anidados que genera pandas. De forma que las 
        tablas tras las operaciones siguen siendo planas (campos e indice simple)
      * Trata al íncide como si fuera un campo más

    Ademas la libreria permite rejecutar y auditar secuencias de transformaciones realizadas previamente
