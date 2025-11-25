;; Modulo para generacion y procesamiento de datos sinteticos
;; Genera puntos 2D con distribucion normal para clasificacion binaria

(ns nn.data
  (:require [nn.model :as model]
            [nn.rng   :as rng]))

;; Genera un numero aleatorio con distribucion normal estandar
;; Usa la transformacion de Box-Muller para convertir uniforme a normal
;; Retorna un valor con media 0 y desviacion estandar 1
(defn rand-normal []
  (let [u (max 1.0e-12 (rng/randd))  ;; evita log(0) con valor minimo
        v (rng/randd)]
    (* (Math/sqrt (* -2.0 (Math/log u)))
       (Math/cos (* 2.0 Math/PI v)))))

;; Genera un numero con distribucion normal personalizada
;; mu: media deseada
;; sigma: desviacion estandar deseada
(defn normal [mu sigma] (+ mu (* sigma (rand-normal))))

;; Genera un punto 2D con distribucion normal alrededor de un centro
;; mu: vector [x y] que representa el centro del cluster
;; sigma: desviacion estandar para ambas dimensiones
(defn make-blob [[mux muy] sigma]
  [(normal mux sigma) (normal muy sigma)])

;; Genera dataset completo con dos clases separadas
;; n: numero de puntos por clase
;; mu1: centro del primer cluster [x y]
;; mu2: centro del segundo cluster [x y]
;; sigma: dispersion de los puntos alrededor del centro
;; Retorna lista de pares [punto etiqueta] donde etiqueta es one-hot
(defn make-blobs [n mu1 mu2 sigma]
  (let [c1 (repeatedly n #(make-blob mu1 sigma))  ;; genera n puntos clase 1
        c2 (repeatedly n #(make-blob mu2 sigma))  ;; genera n puntos clase 2
        y1 (model/one-hot 2 0)  ;; etiqueta clase 1: [1.0 0.0]
        y2 (model/one-hot 2 1)]  ;; etiqueta clase 2: [0.0 1.0]
    (concat (map (fn [x] [(vec x) y1]) c1)  ;; combina puntos con etiquetas
            (map (fn [x] [(vec x) y2]) c2))))

;; Normaliza coordenadas del dataset al rango [0 1] para mejor entrenamiento
;; Esta normalizacion min-max hace que la red converja mas rapido
;; dataset: lista de pares [punto etiqueta]
;; Retorna dataset con coordenadas normalizadas
(defn minmax-2d [dataset]
  (let [xs   (map first dataset)  ;; extrae solo los puntos sin etiquetas
        xs1  (map first xs)  xs2  (map second xs)  ;; separa coordenadas x e y
        min1 (apply min xs1) max1 (apply max xs1)  ;; encuentra rango de x
        min2 (apply min xs2) max2 (apply max xs2)  ;; encuentra rango de y
        scale (fn [v mn mx] (if (== mn mx) 0.5 (/ (- v mn) (- mx mn))))]  ;; funcion de escalado
    (map (fn [[x y]]
           [[(scale (x 0) min1 max1)  ;; normaliza x
             (scale (x 1) min2 max2)]  ;; normaliza y
            y])  ;; mantiene etiqueta sin cambios
         dataset)))
