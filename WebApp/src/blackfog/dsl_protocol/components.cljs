(ns blackfog.dsl-protocol.components
  (:require [clojure.string :as str]
            [reagent.core :as r]
            [blackfog.app-state.core :refer [app-state]]
            [clojure.walk :refer [postwalk]]
            [blackfog.dsl-protocol.core :refer [render-element]]))

(defonce components (r/atom {}))

(defn reg-component [id handler-fn]
  (swap! components assoc id handler-fn))

(defn parser-component [[id & args]]
  (let [handler (get @components id)]
    (render-element (handler args))))

;; 新增递归上下文管理
(defonce recursion-context (r/atom {:max-depth 10 :current 0}))

(defn with-recursion-context [f]
  (when (< (:current @recursion-context) (:max-depth @recursion-context))
    (swap! recursion-context update :current inc)
    (let [result (f)]
      (swap! recursion-context update :current dec)
      result)))

;; 改造后的渲染核心
(defn render [nodes]
  (letfn [(process-node [node]
            (with-recursion-context
              #(if (vector? node)
                 (let [[id & args] node
                       component? (contains? @components id)]
                   (if component?
                     (-> (parser-component node)
                         render)
                     node))
                 node)))]
    (postwalk process-node nodes)))
