(ns blackfog.dsl-protocol.styles
  (:require [clojure.string :as str]
            [blackfog.dsl-protocol.base :refer [reg-element
                                                reg-component
                                                render-element]]))



(reg-element
 :p
 (fn [s] (str/join s)))

(reg-element
 :row
 (fn [s] (str/join "\n" s)))

(reg-element
 :rows (fn [s] (str/join "\n\n" s)))

(reg-element
 :h1
 #(str "# " (str/join %)))

(reg-element
 :h2
 #(str "## " (str/join %)))

(reg-element
 :h3
 #(str "### " (str/join %)))


(reg-element
 :b
 #(str "**" (str/join %) "**"))

(reg-element
 :i
 #(str "__" (str/join %) "__"))

(reg-element
 :del
 #(str "~~" (str/join %) "~~"))

(reg-element
 :block
 (fn [args]
   (str "\n```\n" (str/join "\n" args) "\n```\n")))

(reg-element
 :block/clojure
 (fn [args]
   (str "\n```clojure\n" (str/join "\n" args) "\n```\n")))

(reg-element
 :block/edn
 (fn [args]
   (str "\n```edn\n" (str/join "\n" args) "\n```\n")))

(reg-element
 :commet/md
 (fn [args]
   (str "\n> "(str/join "\n" args) "")))
