;; Modulo principal del modelo de red neuronal
;; Implementa forward propagation, backpropagation y entrenamiento

(ns nn.model
  (:require [nn.math :as m]))

;; Calcula la transformacion lineal de una neurona
;; Formula: z = w·x + b donde w es vector de pesos, x es entrada, b es bias
;; ws: vector de pesos de la neurona
;; b: bias (sesgo) de la neurona
;; x: vector de entrada
;; Retorna valor escalar z (pre-activacion)
(defn affine [ws b x]
  (+ b (m/dot ws x)))

;; Propagacion hacia adelante de una capa completa
;; Calcula z (pre-activacion) y a (activacion) para todas las neuronas
;; layer: mapa con :W (matriz pesos), :b (vector bias), :act (funcion activacion)
;; x: vector de entrada a la capa
;; Retorna mapa con :a (activaciones) y :z (pre-activaciones)
(defn layer-forward [{:keys [W b act]} x]
  (let [z (mapv (fn [ws bi] (affine ws bi x)) W b)  ;; calcula z para cada neurona
        a (mapv act z)]  ;; aplica funcion de activacion a cada z
    {:a a :z z}))

;; Propagacion hacia adelante de toda la red neuronal
;; Va pasando la salida de cada capa como entrada a la siguiente
;; Guarda informacion intermedia (cache) necesaria para backpropagation
;; layers: vector de capas de la red
;; x: vector de entrada a la red
;; Retorna mapa con :a-last (salida final) y :caches (info de cada capa)
(defn network-forward [layers x]
  (loop [a x, caches [], ls layers]  ;; a es la activacion actual, caches guarda historial
    (if (empty? ls)
      {:a-last a :caches caches}  ;; retorna salida final y cache completo
      (let [layer (first ls)
            a-in a  ;; guardamos entrada de esta capa para backprop
            {:keys [a z]} (layer-forward layer a-in)]  ;; calcula salida de la capa
        (recur a  ;; la salida a es entrada de siguiente capa
               (conj caches (merge layer {:a-in a-in :z z}))  ;; guarda info completa
               (rest ls))))))

;; Calcula error cuadratico medio entre prediccion y etiqueta real
;; Formula: MSE = (1/n) * Σ(yhat - y)^2
;; yhat: vector de prediccion
;; y: vector de etiqueta verdadera
;; Retorna escalar con el error promedio
(defn mse [yhat y]
  (let [diffs (map - yhat y)]  ;; calcula diferencias
    (/ (reduce + (map #(* % %) diffs)) (double (count y)))))  ;; promedia cuadrados

;; Encuentra indice del valor maximo en un vector
;; Se usa para convertir probabilidades a clase predicha
;; xs: vector de valores
;; Retorna indice del elemento mas grande
(defn argmax [xs]
  (->> xs (map-indexed vector) (apply max-key second) first))

;; Crea vector one-hot de tamaño n con 1.0 en posicion idx
;; Ejemplo: one-hot(3, 1) = [0.0 1.0 0.0]
;; Se usa para codificar etiquetas de clases
;; n: tamaño del vector
;; idx: posicion donde colocar 1.0
(defn one-hot [n idx]
  (vec (for [i (range n)] (if (= i idx) 1.0 0.0))))

;; Calcula el delta (gradiente local) de la capa de salida
;; Formula: δ = (yhat - y) ⊙ dact(z) donde ⊙ es producto elemento a elemento
;; Este delta se propaga hacia atras en backpropagation
;; a-last: activaciones finales (prediccion)
;; y: etiqueta verdadera
;; z-last: pre-activaciones de salida
;; dact: derivada de funcion de activacion
;; Retorna vector delta de la capa de salida
(defn output-delta [a-last y z-last dact]
  (mapv * (map - a-last y) (map dact z-last)))

;; Transpone una matriz (intercambia filas por columnas)
;; M: matriz representada como vector de filas
;; Retorna matriz transpuesta
(defn transpose [M] (apply mapv vector M))

;; Algoritmo de backpropagation: calcula gradientes para todas las capas
;; Propaga el error desde la salida hacia la entrada
;; Formula clave: δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ dact(z_l)
;; layers: vector de capas de la red
;; x: entrada de ejemplo
;; y: etiqueta verdadera de ejemplo
;; Retorna lista de gradientes {:dW :db} para cada capa
(defn backprop [layers x y]
  (let [{:keys [a-last caches]} (network-forward layers x)  ;; forward pass primero
        last-idx (dec (count caches))]
    (loop [idx last-idx
           delta (output-delta a-last y (:z (nth caches last-idx)) (:dact (nth caches last-idx)))
           grads []]
      (let [{:keys [a-in W]} (nth caches idx)
            ;; Gradiente de pesos: dW = δ ⊗ a_entrada^T (producto externo)
            dW (mapv (fn [delta-neuron]
                       (mapv (fn [ain] (* delta-neuron ain)) a-in))
                     delta)
            ;; Gradiente de bias: db = δ (es directo)
            db delta
            ;; Calcula delta para la capa anterior: δ_prev = (W^T · δ) ⊙ dact(z_prev)
            prev-delta (when (pos? idx)
                         (let [w-trans (transpose W)
                               propagated (mapv (fn [col] (reduce + (map * col delta))) w-trans)
                               prev-cache (nth caches (dec idx))
                               prev-z (:z prev-cache)
                               act-deriv (:dact prev-cache)
                               prev-deriv (mapv act-deriv prev-z)]
                           (mapv * propagated prev-deriv)))]
        (if (zero? idx)
          (conj grads {:dW dW :db db})
          (recur (dec idx) prev-delta (conj grads {:dW dW :db db})))))))

;; Actualiza parametros de la red usando gradiente descendente estocastico
;; Formula: W := W - lr*dW y b := b - lr*db
;; layers: capas actuales de la red
;; grads: gradientes calculados por backprop
;; lr: learning rate (tasa de aprendizaje)
;; Retorna red con parametros actualizados
(defn update-params [layers grads lr]
  (let [grads-in (reverse grads)]  ;; invertimos porque backprop va al reves
    (vec
      (map (fn [{:keys [W b act dact]} {:keys [dW db]}]
             {:W (mapv (fn [rowW rowdW]  ;; actualiza cada fila de pesos
                         (mapv (fn [w dw] (- w (* lr dw))) rowW rowdW))  ;; W - lr*dW
                       W dW)
              :b (mapv (fn [bi dbi] (- bi (* lr dbi))) b db)  ;; b - lr*db
              :act act :dact dact})  ;; mantiene funciones de activacion
           layers grads-in))))

;; Entrena la red por una epoca completa usando mini-batches
;; Una epoca es un pase completo por todos los datos de entrenamiento
;; El mini-batch SGD es mas eficiente que SGD puro o batch completo
;; layers: red neuronal actual
;; data: conjunto de entrenamiento (lista de pares [x y])
;; lr: learning rate
;; batch-size: tamaño del mini-batch
;; Retorna red actualizada despues de procesar todos los batches
(defn train-epoch [layers data lr batch-size]
  (reduce
    (fn [L batch]  ;; L es la red actual, batch es mini-batch de datos
      ;; Inicializa acumuladores de gradientes con ceros (misma forma que W y b)
      (let [init-grads (map (fn [{:keys [W b]}]
                              {:dW (mapv (fn [row] (mapv (constantly 0.0) row)) W)
                               :db (mapv (constantly 0.0) b)})
                            L)
            ;; Suma gradientes de todos los ejemplos del batch
            sum-grads  (reduce
                         (fn [acc [x y]]  ;; para cada ejemplo en el batch
                           (let [gs (backprop L x y)]  ;; calcula gradientes
                             (map (fn [{aW :dW aB :db} {bW :dW bB :db}]  ;; suma con acumulador
                                    {:dW (mapv (fn [ra rb] (mapv + ra rb)) aW bW)
                                     :db (mapv + aB bB)})
                                  acc gs)))
                         init-grads
                         batch)
            ;; Calcula promedio de gradientes del batch
            m (double (count batch))  ;; numero de ejemplos en batch
            avg-grads (map (fn [{:keys [dW db]}]
                             {:dW (mapv (fn [row] (mapv #(/ % m) row)) dW)  ;; dW/m
                              :db (mapv #(/ % m) db)})  ;; db/m
                           sum-grads)]
        (update-params L avg-grads lr)))  ;; actualiza pesos con gradiente promedio
    layers
    (partition-all batch-size (shuffle data))))  ;; divide datos en batches y mezcla

;; Calcula exactitud (accuracy) de la red en un conjunto de datos
;; Exactitud = proporcion de predicciones correctas
;; layers: red neuronal
;; dataset: conjunto de datos de evaluacion
;; Retorna valor entre 0 y 1 (porcentaje de aciertos)
(defn accuracy [layers dataset]
  (let [hits (for [[x y] dataset
                   :let [yhat (:a-last (network-forward layers x))]]  ;; predice cada ejemplo
               (if (= (argmax yhat) (argmax y)) 1 0))]  ;; compara clase predicha con real
    (/ (reduce + hits) (double (count dataset)))))  ;; promedia aciertos

;; Hace prediccion para un solo ejemplo
;; layers: red neuronal
;; x: vector de entrada
;; Retorna vector de probabilidades (una por clase)
(defn predict [layers x]
  (:a-last (network-forward layers x)))

;; Calcula perdida MSE promedio en todo un conjunto de datos
;; Util para monitorear que tan bien ajusta la red
;; layers: red neuronal
;; dataset: conjunto de datos
;; Retorna error cuadratico medio promedio
(defn loss-dataset [layers dataset]
  (let [losses (map (fn [[x y]] (mse (predict layers x) y)) dataset)]  ;; MSE por ejemplo
    (/ (reduce + losses) (double (count losses)))))  ;; promedia todos los MSE
