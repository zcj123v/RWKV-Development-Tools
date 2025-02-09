(ns blackfog.dsl-protocol.impl
  (:require [clojure.string :as str]))

(defn is-form? [form]
  (and (vector? form)
       (not (map-entry? form))
       (let [f (first form)]
         (or (keyword? f)
             (symbol? f)
             (fn? f)))))


(defn- prompt-transform [form]
  (if (is-form? form)
    (let [[tag & children] form
          ;; 新增命名空间处理逻辑
          tag-symbol (if-let [ns (namespace tag)] ; 提取关键字的命名空间
                       (symbol ns (name tag))     ; 带命名空间的符号
                       (symbol (name tag)))]      ; 普通符号
      `(~tag-symbol ~@(map prompt-transform children)))
    form))


(defmacro prompt [form]
  (prompt-transform form))

(defmacro defprompt [sym form]
  `(def ~sym ~(prompt-transform form)))

(defmacro defprompt-dynamic [sym form]
  `(def ~sym (fn [& args#]
               (-> ~(prompt-transform form)
                   (concat args#)
                   flatten))))

(defmacro defhc [register-state key form]
  `(swap! ~register-state assoc ~key ~(prompt-transform form)))


(comment
  (def is-form? #(and (vector? %)
                      (not (map-entry? %))
                      (or (keyword? (first %))
                          (symbol? (first %))
                          (fn? (first %)))))
  (defn- prompt-transform [form]
    (if (vector? form)
      (let [[tag & children] form]
        `(~(symbol (name tag)) ~@(map prompt-transform children)))
      form)))
