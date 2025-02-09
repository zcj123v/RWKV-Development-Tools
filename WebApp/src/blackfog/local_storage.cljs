(ns blackfog.local-storage
  (:require [cljs.reader :as reader]))

;; 辅助函数：将 Clojure 数据结构序列化为字符串
(defn serialize-data
  "将数据序列化为可存储的字符串"
  [data]
  (pr-str data))

;; 辅助函数：反序列化数据
(defn deserialize-data
  "从字符串反序列化数据"
  [data-str]
  (clojure.edn/read-string data-str))

(defn get-all-storage-keys
  "获取所有 LocalStorage 的键"
  []
  (let [keys (js/Object.keys js/localStorage)]
    (vec keys)))

;; 保存数据到 LocalStorage
(defn save-to-storage!
  "将数据保存到 LocalStorage"
  [key data]
  (.setItem js/localStorage key (serialize-data data)))


;; 从 LocalStorage 读取数据
(defn load-from-storage
  "从 LocalStorage 读取数据"
  [key]
  (let [item (.getItem js/localStorage key)]
    (when item
      (deserialize-data item))))

;; 删除 LocalStorage 中的特定项
(defn remove-from-storage!
  "从 LocalStorage 删除指定 key 的数据"
  [key]
  (.removeItem js/localStorage key))

;; 清空 LocalStorage
(defn clear-storage!
  "清空 LocalStorage 中的所有数据"
  []
  (.clear js/localStorage))

;; 使用示例
(defn save-user-preferences!
  "保存用户偏好设置"
  [preferences]
  (save-to-storage! "user-preferences" preferences))

(defn load-user-preferences
  "加载用户偏好设置"
  []
  (load-from-storage "user-preferences"))

(defn get-all-local-storage
  "获取 LocalStorage 中的所有键值对"
  []
  (let [local-storage js/localStorage
        keys (range (.-length local-storage))]
    (->> keys
         (map (fn [index]
                (let [key (.key local-storage index)
                      value (.getItem local-storage key)]
                  [key value])))
         (into {}))))

;; 使用示例
(defn log-storage-contents []
  (let [storage-map (get-all-local-storage)]
    (println "本地存储内容:" storage-map)))

;; 带有反序列化的高级版本
(defn get-all-local-storage-with-parsing
  "获取并尝试解析 LocalStorage 中的所有键值对"
  ([]
   (get-all-local-storage-with-parsing
    (fn [v]
      (try
        (reader/read-string v)
        (catch :default _ v)))))

  ([parse-fn]
   (let [local-storage js/localStorage
         keys (range (.-length local-storage))]
     (->> keys
          (map (fn [index]
                 (let [key (.key local-storage index)
                       value (.getItem local-storage key)
                       parsed-value (parse-fn value)]
                   [key parsed-value])))
          (into {})))))

;; 过滤和处理存储内容的示例
(defn get-filtered-storage
  "获取特定前缀或类型的存储项"
  [& {:keys [prefix-filter
             type-filter]
      :or {prefix-filter (constantly true)
           type-filter (constantly true)}}]
  (let [storage-map (get-all-local-storage-with-parsing)]
    (->> storage-map
         (filter (fn [[k v]]
                   (and
                    (prefix-filter k)
                    (type-filter v))))
         (into {}))))

;; 使用示例
(comment
  ;; 获取所有存储项
  (get-all-local-storage)

  ;; 获取所有以 "user-" 开头的项
  (get-filtered-storage
   :prefix-filter #(str/starts-with? % "user-"))

  ;; 获取所有映射类型的项
  (get-filtered-storage
   :type-filter map?)

  ;; 组合使用
  (get-filtered-storage
   :prefix-filter #(str/starts-with? % "user-")
   :type-filter map?))

;; 安全地清理特定类型的存储项
(defn clear-storage-by-type!
  "按类型清理存储项"
  [type-filter]
  (let [to-remove (get-filtered-storage
                   :type-filter type-filter)]
    (doseq [[k _] to-remove]
      (.removeItem js/localStorage k))))
