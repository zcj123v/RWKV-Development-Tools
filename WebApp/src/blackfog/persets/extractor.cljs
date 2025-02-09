(ns blackfog.persets.extractor
  (:require [clojure.string :as str]))


(defn think->inner [text]
  (try (->> (re-seq #"<think>([^<]*(?:<(?!/think>)[^<]*)*)</think>" text)
            (map second)
            (str/join "\n"))
       (catch js/Error e
         (println "wrong-->" (.-message e)))))

(defn think->outer [text]
  (try (-> (str/replace text #"<think>[\s\S]*?</think>"  "\n")
           (str/replace  #"<.?think>"  ""))

       (catch js/Error e
         (println "wrong-->" (.-message e)))))

(defn iphone->inner [text]
  (try
    (->> (re-seq #"<iphone>\n([\s\S]*?)\n</iphone>" text)
         (map second)
         (str/join))
    (catch js/Error e
      (println "wrong-->" (.-message e)))))

(defn iphone->inner-edn [content]
  (try (let [reality-pattern #"<iphone>\n([\s\S]*?)\n</iphone>"]
         (->> (re-seq reality-pattern content)
              (map second)
              (map clojure.edn/read-string)))
       (catch js/Error e
         (println "wrong-->" (.-message e)))))

(defn xml->map [text]
  (try (let [pattern #"<(.*?)>([\s\S]*?)</\1>"
             coll (re-seq pattern text)
             coll  (map rest coll)]
         (reduce (fn [m [k v]] (assoc m (keyword  k)
                                      (str/trim v))) {} coll ))
       (catch js/Error e
         (println "wrong-->" (.-message e)))))

(defn markdown [content]
  (try (let [pattern #"```markdown\n([\s\S]*?)\n```"
             coll (into [] (comp (map last)) (re-seq pattern content))]
         (clojure.string/join "\n\n" coll))
       (catch js/Error e
         (println "wrong-->" (.-message e)))))


(defn codeblock-edn [content]
  (try (let [pattern #"```edn\n([\s\S]*?)\n```"
             coll (into [] (comp (map last)
                                 (mapcat cljs.reader/read-string))
                        (re-seq pattern content))]
         coll)
       (catch js/Error e
         (println "wrong-->" (.-message e)))))

(defn markdown->edn [markdown-text]
  (let [entity-type-pattern #"(?m)^#\s+(.+?)$\n((?:(?!^#\s+).+\n?)*)"
        entity-pattern #"(?m)^##\s+(.+?)$\n((?:(?!^##?\s+).+\n?)*)"
        prop-pattern #"(?m)^###\s+(.+?)(?::\s*|\n)([^#].+?)(?=\n###|\n##|\n#|$)"]
    (for [[_ type-name type-content] (re-seq entity-type-pattern markdown-text)]
      {(keyword type-name)
       (for [[_ entity-name entity-content] (re-seq entity-pattern type-content)]
         {(keyword entity-name)
          (into {}
                (for [[_ prop-name prop-value] (re-seq prop-pattern entity-content)]
                  [(keyword prop-name) (str/trim prop-value)]))})})))


(defn text->pure-text [text]
  (str/replace text #"\\\\u([0-9a-fA-F]{4})"
               (fn [[_ hex]] (js/String.  (js/parseInt hex 16)))))
