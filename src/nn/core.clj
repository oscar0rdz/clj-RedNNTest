;; Archivo principal del proyecto de red neuronal
;; Aqui definimos la arquitectura de la red y el proceso de entrenamiento

(ns nn.core
  (:require [nn.math  :as m]
            [nn.model :as model]
            [nn.data  :as data]
            [nn.rng   :as rng]
            [nn.visualization :as viz])
  
  (:import [java.io File]))

;; Funcion para crear matriz de pesos aleatoria
;; out: numero de neuronas de salida
;; in: numero de neuronas de entrada
;; Retorna un vector de vectores con valores entre -0.5 y 0.5
(defn rand-w [out in]
  (vec (repeatedly out (fn [] (vec (repeatedly in rng/randm05))))))

;; Funcion para crear vector de bias aleatorio
;; out: numero de neuronas
;; Retorna un vector con valores entre -0.5 y 0.5
(defn rand-b [out]
  (vec (repeatedly out rng/randm05)))

;; Funcion que construye la arquitectura de la red neuronal
;; Arquitectura: 2 entradas -> 8 neuronas ocultas -> 2 salidas
;; Usa funcion sigmoide como activacion en ambas capas
;; Retorna un vector de mapas donde cada mapa representa una capa
(defn make-network []
  (let [act  m/sigmoid    ;; funcion de activacion
        dact m/dsigmoid]  ;; derivada de la funcion de activacion
    [{:W (rand-w 8 2) :b (rand-b 8) :act act :dact dact}  ;; capa oculta
     {:W (rand-w 2 8) :b (rand-b 2) :act act :dact dact}])) ;; capa de salida

;; Funcion principal de entrenamiento
;; epochs: numero de iteraciones completas sobre los datos
;; lr: learning rate o tasa de aprendizaje
;; batch-size: tamaño del lote para gradiente descendente
;; n-per-class: numero de puntos por clase a generar
(defn train [epochs lr batch-size n-per-class]
  (let [raw     (data/make-blobs n-per-class [0.2 0.2] [0.8 0.8] 0.08)  ;; genera datos sinteticos
        dataset (shuffle (vec (data/minmax-2d raw)))  ;; normaliza y mezcla
        split   (int (Math/floor (* 0.8 (count dataset))))  ;; calcula punto de division 80/20
        train   (subvec dataset 0 split)  ;; conjunto de entrenamiento
        test    (subvec dataset split (count dataset))]  ;; conjunto de prueba
    ;; loop es recursion con nombres: ep es el contador de epoca, net es la red actual
    (loop [ep 1, net (make-network)]
      (if (> ep epochs)
        ;; cuando terminamos todas las epocas mostramos resultados finales
        (do
          (println "Entrenamiento terminado.")
          (println "Accuracy (train):" (format "%.3f" (model/accuracy net train)))
          (println "Accuracy (test): " (format "%.3f" (model/accuracy net test)))
          {:net net :test test})
        ;; en cada epoca entrenamos la red y mostramos metricas
        (let [net' (model/train-epoch net train lr batch-size)  ;; entrena una epoca
              acc  (model/accuracy net' test)  ;; calcula exactitud en test
              lss  (model/loss-dataset net' test)]  ;; calcula perdida en test
          (println "Época" ep "| acc(test)=" (format "%.3f" acc)
                           "| mse(test)="  (format "%.4f" lss))
          (recur (inc ep) net'))))))  ;; siguiente epoca con red actualizada

;; Punto de entrada del programa
;; Entrena una red con 20 epocas, lr 0.5, batch 16, 400 puntos por clase
;; Genera visualizacion SVG de la frontera de decision
(defn -main [& _]
  (let [{:keys [net test]} (train 20 0.5 16 400)
        svg-content (viz/generate-svg net test)]
    (.mkdirs (File. "images"))
    (spit "images/decision_boundary.svg" svg-content)
    (println "\nVisualización guardada en images/decision_boundary.svg")))


