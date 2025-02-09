(ns blackfog.utils.message-tools
  (:require ["crypto-js" :as crypto]))

(defn text->128bits [text]
  (-> (crypto/MD5 text)
      .toString
      (subs 0, 32)))

(def add-message [messages msg]
  (conj (into [] messages)
        (assoc msg :id (text->128bits (:content msg)))))

(defn delete-message [messages id]
  (vec (remove #(= (:id %) id) messages)))

(defn update-message [messages id new-content]
  (vec (map #(if (= (:id %) id)
               (assoc % :content new-content
                      :timestamp (js/Date.now))
               %)
            messages)))

(defn find-message-by-id [messages id]
  (first (filter #(= (:id %) id) messages)))

(defn insert-message-at [messages index msg]
  (let [msg-with-id (assoc msg :id (text->128bits (:content msg)))]
    (vec (concat (subvec messages 0 index)
                 [msg-with-id]
                 (subvec messages index)))))

(defn search-messages [messages query]
  (filter #(clojure.string/includes?
            (clojure.string/lower-case (:content %))
            (clojure.string/lower-case query))
          messages))

;; 1. 切片功能（时空分形）
(defn slice-messages [messages start-idx end-idx]
  (let [max-idx (count messages)
        safe-end (min end-idx max-idx)]
    (subvec (vec messages) (max 0 start-idx) safe-end)))

;; 2. 量子态插入（在指定位置前注入记忆碎片）
(defn insert-slice-before [messages target-idx slice]
  (let [safe-idx (-> target-idx (max 0) (min (count messages)))
        with-ids (mapv #(assoc % :id (text->128bits (:content %))) slice)]
    (vec (concat (subvec messages 0 safe-idx)
                 with-ids
                 (subvec messages safe-idx)))))

;; 3. 末梢神经改造术（实时流式污染）
(defn update-last-message-by-stream [messages new-content]
  (if-let [last-msg (peek messages)]
    (conj (pop messages)
          (-> last-msg
              (assoc :content (str (:content last-msg) new-content))
              (assoc :timestamp (js/Date.now))))
    messages))

(defn streaming-update-at [messages idx new-fragment]
  (if-let [target (get messages idx)]
    (let [updated (update target :content #(str % new-fragment))]
      (assoc messages idx (assoc updated :timestamp (js/Date.now))))
    messages))

;; 增强版（带缓冲池的末梢快感累积）
(defn buffered-streaming-update [messages idx fragment]
  (let [buffer-key (keyword (str "buffer-" idx))]
    (-> messages
        (update-in [idx :content] #(str % (get % buffer-key "") fragment))
        (assoc-in [idx buffer-key] "")  ;; 清空缓冲区
        (update-in [idx :content] #(str % fragment))  ;; 直接追加
        (assoc-in [idx :timestamp] (js/Date.now)))))
