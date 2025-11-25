;; Modulo responsable de generar una visualización SVG de la red
;; Tiene helpers para dibujar la frontera de decisión y los datos reales

(ns nn.visualization
  (:require [clojure.string :as str]
            [nn.model :as model]))

(def width 640.0)
(def height 640.0)
(def margin 40.0)
(def grid-resolution 70)
(def plot-size (- width (* 2 margin)))
(def cell-size (/ plot-size grid-resolution))
(def class-colors
  {0 {:fill "#1f77b4" :stroke "#0a2c4a"}
   1 {:fill "#ff7f0e" :stroke "#7c3d06"}})
(def base-color-0 [31 119 180])
(def base-color-1 [255 127 14])

(defn clamp [x] (-> x (max 0.0) (min 1.0)))

(defn mix-channel [a b t]
  (int (Math/round (+ a (* (- b a) t)))))

(defn mix-colors [t]
  (let [t' (clamp t)]
    (format "rgb(%d,%d,%d)"
            (mix-channel (base-color-0 0) (base-color-1 0) t')
            (mix-channel (base-color-0 1) (base-color-1 1) t')
            (mix-channel (base-color-0 2) (base-color-1 2) t'))))

(defn x->px [x] (+ margin (* x plot-size)))
(defn y->px [y] (- (+ margin plot-size) (* y plot-size)))

(defn decision-cells [net]
  (for [i (range grid-resolution)
        j (range grid-resolution)
        :let [nx (/ (+ 0.5 i) grid-resolution)
              ny (/ (+ 0.5 j) grid-resolution)
              probs (model/predict net [nx ny])
              prob-class1 (nth probs 1)
              color (mix-colors prob-class1)
              cx (x->px nx)
              cy (y->px ny)
              x (- cx (/ cell-size 2.0))
              y (- cy (/ cell-size 2.0))]]
    (format "<rect x='%.2f' y='%.2f' width='%.2f' height='%.2f' fill='%s' fill-opacity='0.45' stroke='none'/>"
            x y cell-size cell-size color)))

(defn point-circles [test-data]
  (for [[pt label] test-data
        :let [cls (model/argmax label)
              {:keys [fill stroke]} (class-colors cls)
              [px py] [(x->px (pt 0)) (y->px (pt 1))]]]
    (format "<circle cx='%.2f' cy='%.2f' r='4.2' fill='%s' stroke='%s' stroke-width='1.5' />"
            px py fill stroke)))

(defn legend []
  (let [entries [{:label "Clase 0" :colors (class-colors 0)}
                 {:label "Clase 1" :colors (class-colors 1)}]]
    (for [[idx {:keys [label colors]}] (map-indexed vector entries)
          :let [y (+ margin (* idx 22.0))
                legend-x (+ margin plot-size -150.0)]]
      (format "<g transform='translate(%.1f,%.1f)'><rect width='14' height='14' rx='3' fill='%s' stroke='%s' stroke-width='1.5'/><text x='22' y='12' font-size='14' fill='#1f1f1f' font-family='monospace'>%s</text></g>"
              legend-x y (:fill colors) (:stroke colors) label))))

(defn generate-svg [net test-data]
  (let [background (str "<rect width='" width "' height='" height "' fill='#f9fafc'/>")
        title (format "<text x='%.1f' y='28' font-size='20' font-family='monospace' fill='#111111'>Frontera de decisión (test set)</text>"
                      margin)
        grid (decision-cells net)
        points (point-circles test-data)
        legend-items (legend)]
    (str/join
      "\n"
      (concat [(format "<svg xmlns='http://www.w3.org/2000/svg' width='%.0f' height='%.0f' viewBox='0 0 %.0f %.0f'>"
                       width height width height)
               background
               title
               (format "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='none' stroke='#b6bdc7' stroke-width='1.5'/>"
                       margin margin plot-size plot-size)]
              grid
              points
              legend-items
              ["</svg>"]))))
