;; Modulo de operaciones matematicas para redes neuronales
;; Contiene algebra lineal basica y funciones de activacion

(ns nn.math)

;; Producto punto entre dos vectores
;; Calcula suma de productos elemento a elemento: x1*y1 + x2*y2 + ...
;; Es la operacion fundamental para calcular w·x + b en neuronas
;; xs: primer vector
;; ys: segundo vector
;; Retorna escalar resultado del producto punto
(defn dot [xs ys]
  (reduce + (map * xs ys)))

;; Suma de dos vectores elemento a elemento
;; [a b c] + [x y z] = [a+x b+y c+z]
(defn add [xs ys] (mapv + xs ys))

;; Resta de dos vectores elemento a elemento
;; [a b c] - [x y z] = [a-x b-y c-z]
(defn sub [xs ys] (mapv - xs ys))

;; Multiplicacion de vector por escalar
;; c * [a b c] = [c*a c*b c*c]
;; c: escalar
;; xs: vector
(defn smul [c xs] (mapv (fn [x] (* c x)) xs))

;; Funcion sigmoide: convierte cualquier numero a rango (0 1)
;; Formula: σ(z) = 1 / (1 + e^(-z))
;; Se usa como funcion de activacion en neuronas
;; z: valor de entrada (pre-activacion)
;; Retorna valor entre 0 y 1
(defn sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

;; Derivada de la funcion sigmoide
;; Formula: σ'(z) = σ(z) * (1 - σ(z))
;; Se usa en backpropagation para calcular gradientes
;; z: valor de entrada
;; Retorna derivada evaluada en z
(defn dsigmoid [z] (let [s (sigmoid z)] (* s (- 1.0 s))))

;; Funcion ReLU: Rectified Linear Unit
;; Formula: ReLU(z) = max(0, z)
;; Activacion alternativa que es mas rapida de calcular
(defn relu [z] (max 0.0 z))

;; Derivada de ReLU
;; Es 1 si z > 0, sino es 0
;; Se usa en backpropagation cuando ReLU es la activacion
(defn drelu [z] (if (pos? z) 1.0 0.0))

;; Aplica una funcion a cada elemento de un vector
;; Utilidad de orden superior para transformar vectores
;; f: funcion a aplicar
;; xs: vector de entrada
(defn vmap [f xs] (mapv f xs))
