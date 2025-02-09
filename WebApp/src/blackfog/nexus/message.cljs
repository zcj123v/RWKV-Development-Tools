(ns blackfog.nexus.message
  (:require
   [cljs.core.async :refer [go chan <! >! go-loop put!] :as async]
   [ajax.core :refer [POST]]
   [clojure.string :as str]))


(defprotocol MessageNode
  (id [this] "获取id")
  (msg->map [this] "解析为map")
  (metadata [this] "获取metadata")
  (update-metadata [this new-meta] "更新metadata")
  (msg->text [this nexus] "解析为文本，结合角色"))

(defrecord Message [id role content timestamp metadata]
  MessageNode

  (id [this] (:id this))

  (msg->map [this] (into {} (seq this)))

  (metadata [this] (:metadata this))

  (update-metadata [this new-meta]
    (update this :metadata merge new-meta))

  (msg->text [this nexus]
    (let [role-name (get-in (deref (:state nexus)) [:personas :personas/name])
          role-name (cond (keyword? role-name) (name role-name)
                          (string? role-name) role-name
                          :always (str role-name))
          {:keys [role content]} this]
      (str "**" role-name "**: " (str/trim content)))))



;; base protocol
(defprotocol BaseNexusPoint
  "定义通信节点的基本行为"
  (get-state [this] "获取状态")
  (get-data [this] "数据")
  (history [this] "获取历史消息记录")
  (reset-history! [this new-history] "存储历史记录")
  (cache [this] "获取缓存内容")
  (output [this] "获取输出内容")
  (clear-cache! [this] "清理缓存")
  (last-output [this] "获取最后一次输出")
  (last-message-with-meta  [this new-meta] "更新最后一条消息的元数据槽"))

(defprotocol StorageOnNexusPoint
  (save-msg [this] "保存信息")
  (load-msg [this] "读取信息")
  (clean-msg [this] "清除信息"))

(defprotocol MessageInNexusPoint
  "定义message操作类型"
  (create-message [this role content] "创造一个新记录")
  (add-message [this message] "添加")
  (remove-message [this message-id] "移除")
  (update-message [this message-id message] "更新")
  (last-message [this] "获取最后一条消息"))

;; 历史信息修改
(defprotocol HistoryOnNexusPoint
  "历史信息管理"
  (set-message [this row-id content] "修改message")
  (find-by [this row-id] "查找信息"))

(defn find-by-message-handler [this state id]
  (get-in @state [:history id]))

(defn set-message-handler [this state id content]
  (swap! state assoc-in [:history id :content] content))

;; batch op
(defprotocol BatchDeleteOperations
  (delete-range [this start end] "删除一个范围内的消息")
  (delete-by-pred [this pred] "根据谓词删除消息"))

(defn normalize-range
  "标准化范围，确保：
   1. start <= end
   2. 范围在合法边界内
   3. 处理负数索引（类似Python的切片）"
  [start end coll-size]
  (let [normalize-idx (fn [idx]
                        (cond
                          (neg? idx) (max 0 (+ coll-size idx))
                          (>= idx coll-size) (dec coll-size)
                          :else idx))
        start' (normalize-idx start)
        end'   (normalize-idx end)]
    [(min start' end') (max start' end')]))

(defn delete-handler [this index]
  (let [messages (history this)]
    (when (and (>= index 0) (< index (count messages)))
      (swap! (get-state this) update :history
             (fn [hist]
               (vec (keep-indexed
                     (fn [idx item]
                       (when (not= idx index) item))
                     hist)))))))

(defn delete-range-handler [this state start end]
  (let [hist (history this)
        size (count hist)]
    (when (pos? size)
      (let [[norm-start norm-end] (normalize-range start end size)]
        (swap! state update :history
               (fn [hist]
                 (->> hist
                      (keep-indexed
                       (fn [idx item]
                         (when (or (< idx norm-start)
                                   (> idx norm-end))
                           item)))
                      vec)))))))
