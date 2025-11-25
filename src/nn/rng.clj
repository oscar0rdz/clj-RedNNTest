;; Modulo de generacion de numeros aleatorios
;; Usa semilla fija para reproducibilidad de experimentos

(ns nn.rng)

;; Generador de numeros aleatorios con semilla fija 1337
;; ^:dynamic permite redefinir en tiempo de ejecucion si es necesario
;; Usar semilla fija garantiza mismos resultados en cada ejecucion
(def ^:dynamic *rng* (java.util.Random. 1337))

;; Genera numero aleatorio en rango [0, 1)
;; Distribucion uniforme
(defn randd [] (.nextDouble *rng*))

;; Genera numero aleatorio en rango (-0.5, 0.5)
;; Util para inicializacion de pesos centrada en cero
(defn randm05 [] (- (randd) 0.5))
