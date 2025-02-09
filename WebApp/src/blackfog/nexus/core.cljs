(ns blackfog.nexus.core
  (:require
   [cljs.core.async :refer [go chan <! >! go-loop put!] :as async]
   [datascript.core :as d]
   [ajax.core :refer [POST]]
   [blackfog.dsl.core :as dsl]
   [blackfog.app-state.core :as app-state]
   [blackfog.api :as raw-api]
   [blackfog.nexus.personas-data :as ps-data]
   [blackfog.nexus.api :as api]

   [blackfog.nexus.message :as msg :refer [->Message]]
   [blackfog.nexus.personas :as ps]
   [blackfog.persets.personas.core]
   [blackfog.local-storage :refer [save-to-storage!
                                   load-from-storage
                                   remove-from-storage!
                                   clear-storage!]]))

;; nexus-point
(defrecord NexusPoint [state]

  ;; -------- base protocol --------
  msg/BaseNexusPoint
  (msg/get-state [this]
    state)

  (msg/get-data [this]
    (deref state))

  (msg/history [this]
    (-> this :state deref :history vec))

  (msg/reset-history! [this new-history]
    (do (swap! state assoc :history (vec new-history))
        (msg/save-msg this)))

  (msg/cache [this]
    (get-in @state [:cache] ""))

  (msg/output [this]
    (get-in @state [:output] ""))

  (msg/clear-cache! [this]
    (swap! state assoc :cache ""))

  (msg/last-output [this]
    (get @state :output))

  (msg/last-message-with-meta [this new-meta]
    (let [history (msg/history this)
          head    (vec (drop-last history))
          tail    (last history)
          tail    (msg/update-metadata tail new-meta)
          history (conj head tail)]
      (msg/reset-history! this history)))

  ;; -------- --------
  msg/StorageOnNexusPoint

  (msg/load-msg [this]
    (swap! state assoc :history
           (into [] (map msg/map->Message)
                 (load-from-storage (ps-data/personas-name this)))))
  (msg/save-msg [this]
    (let [history (msg/history this)]
      (save-to-storage!
       (ps-data/personas-name this)
       (into [] (map #(msg/msg->map %) history)))))

  (msg/clean-msg [this]
    (let [history (msg/history this)]
      (msg/reset-history! this [])
      (remove-from-storage! (ps-data/personas-name this))))

  ;; -------- --------
  msg/MessageInNexusPoint

  (msg/create-message [this role content]
    (->Message (str (d/squuid))  role content (js/Date.) {}))

  (msg/add-message [this message]
    (let [history (msg/history this)
          history (conj history message)]
      (msg/reset-history! this history)))

  (msg/remove-message [this message-id]
    (let [history (msg/history this)
          history (remove #(= (:id %) message-id) history)]
      (msg/reset-history! this history)))

  (msg/update-message [this message-id new-msg]
    (let [history (msg/history this)
          f (fn [m n] (if (= (:id n) message-id)
                        (conj m (merge n new-msg))
                        (conj m n)))
          history (reduce f [] history)]
      (msg/reset-history! this history)))

  (msg/last-message [this]
    (last (msg/history this)))

  ;; -------- --------
  msg/HistoryOnNexusPoint
  (msg/set-message [this id content]
    (msg/set-message-handler this state id content)
    (msg/save-msg this))

  (msg/find-by [this id]
    (msg/find-by-message-handler this state id))

  ;; -------- --------
  msg/BatchDeleteOperations
  (delete-range [this start end]
    (msg/delete-range-handler this state start end))

  (delete-by-pred [this pred]
    (swap! state update :history
           (fn [hist] (vec (remove pred hist)))))

  ;; -------- callback protocol --------
  api/CallBackNexusPoint
  (api/send! [this msg callback ]
    (api/callback-send-handler this state msg callback ))

  ;; -------- 实现流式处理协议 --------
  api/StreamingNexusPoint
  (api/stream! [this config msg callback]
    (api/stream-send-handler this state config msg callback))

  (api/append-cache! [this content]
    (swap! state update :cache str content))

  (api/flush-cache! [this msg postfix-fn]
    (api/stream-flush-cache this state msg postfix-fn))

  ;; -------- 初始化人格，接受后处理、中间件 --------
  ps-data/BasePersonasNexusPoint

  (ps-data/init-personas [this personas-key]
    (ps-data/init-personas-handler this state personas-key))

  (ps-data/personas-name [this]
    (str (get-in @state [:personas :personas/name] "default")))

  (ps-data/postfix-in-receive [this]
    (get-in @state [:personas :recive.fn/postfix] identity))

  (ps-data/middlewares-in-receive [this]
    (get-in @state [:personas :recive.fn/middlewares] []))

  ;; -------- 信息接受协议 --------
  ps/ReciveFromPhysicWorldByNexusPoint

  (ps/prepare-for-receive [this personas  user-input]
    "准备接受信息的对话模板")

  (ps/recive! [this user-input callback]
    (ps/receive-handler this state  user-input callback)))


;; 工厂函数
(defn create-stream-nexus []
  (->NexusPoint (atom {:history []
                       :cache ""
                       :output nil})))

(def t-atom (atom {:history []
                   :cache ""
                   :output nil}))

(comment
  (let [data (->NexusPoint t-atom)]
    (ps/init-personas data :Eve)
    (msg/get-data data)
    (ps/recive! data :Eve "你好啊"))
  (println @t-atom))


;; 使用示例
(comment
  (let [data (->NexusPoint test-atom)]
    (api/stream! data "你好啊" println)
    )
  (let [nexus (create-stream-nexus)
        handle-chunk (fn [chunk]
                       (if (= chunk :done)
                         (println "Stream completed!")
                         (println "Received chunk:" chunk)))]
    (api/stream! nexus "Hello, stream!" handle-chunk)))
