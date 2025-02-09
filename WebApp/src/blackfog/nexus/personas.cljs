(ns blackfog.nexus.personas
  (:require
   [cljs.core.async :refer [go chan <! >! go-loop put!] :as async]
   [ajax.core :refer [POST]]
   [blackfog.nexus.message :as msg]
   [blackfog.dsl.core :as dsl]
   [blackfog.api :as bf-api]
   [blackfog.app-state.core :as app-state]
   [blackfog.nexus.api :as api]
   [blackfog.persets.personas.common :refer [personas-preset]]
   [blackfog.persets.personas.the-reader :refer [TheReaderPrefix]]
   [blackfog.persets.services :refer [services]]))


(defprotocol PeronsasMemoryInNexusPoint
  "定义通信节点的基本行为"
  (recall [this] "回忆"))

(defprotocol ContactByNexusPoint
  "定义通信节点的基本行为"
  (need-to-speak? [this personas] "检查是否应当发言？")
  (prepare-for-speak [this personas]  "准备接受信息的对话模板")
  (speak [this personas]  "向外部发出消息")
  (need-to-snap? [this personas] "检查是否应当小憩？")
  (prepare-for-snap [this personas]  "准备接受信息的对话模板")
  (snap  [this personas]   "小憩，整理对话，构建短期记忆"))


(defprotocol ReciveFromPhysicWorldByNexusPoint
  "定义通信节点的基本行为"
  (prepare-for-receive [this personas  user-input]  "准备接受信息的对话模板")
  (recive! [this user-input callback]    "接收到信息"))

(defn receive-handler
  [this state  user-input callback]
  (let [recive-fn (get-in @state [:personas :recive])
        user-prefix-fn (get-in @state [:personas :prefix/reader]
                               (fn [turn content] (str "turns: " turn "\n"
                                                       content)))
        history    (msg/history this)
        user-input ( user-prefix-fn  (int (count history)) user-input )
        history   (take-last 16 history)
        msg       (recive-fn history user-input)
        config-key (get-in @state [:personas :service/config])
        config    (get services config-key (:default services))]
    (api/stream! this config msg callback)))
