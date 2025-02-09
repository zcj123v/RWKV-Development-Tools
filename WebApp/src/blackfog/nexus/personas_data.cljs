(ns blackfog.nexus.personas-data
  (:require [blackfog.nexus.message :as msg]
            [blackfog.persets.personas.common :refer [personas-preset]]
            [blackfog.persets.services :refer [services]]))

(defprotocol BasePersonasNexusPoint
  "定义通信节点的基本行为"
  (init-personas [this personas-key] "初始化人格")
  (personas-name [this] "人格名称")
  (postfix-in-receive     [this] "预处理")
  (middlewares-in-receive [this] "中间件"))

(defn init-personas-handler [this state personas-key
                             & {:keys [preset]
                                :or {preset personas-preset}}]
  (swap! state assoc :personas (get personas-preset personas-key
                                    (:default personas-preset))
         :history (into [] (get @state :history))))
