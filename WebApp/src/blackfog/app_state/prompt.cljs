(ns blackfog.app-state.prompt
  (:require [reagent.core :as r]))

(defonce prompt-space
  (r/aom {:fx-handlers {}
          :event-handlers {}
          :subscriptions {}}))

;; 事件系统
(defn reg-event
  "注册事件处理器
   id: 事件标识符
   handler-fn: (fn [state event] new-state)"
  [id handler-fn]
  (swap! prompt-space assoc-in [:event-handlers id] handler-fn))

;; 效果系统
(defn reg-fx
  "注册效果处理器
   id: 效果标识符
   handler-fn: (fn [coeffects event] effects)"
  [id handler-fn]
  (swap! prompt-space assoc-in [:fx-handlers id] handler-fn))

;; 订阅系统
(defn reg-sub
  "注册订阅
   id: 订阅标识符
   compute-fn: (fn [state query-vector] result)"
  [id compute-fn]
  (swap! prompt-space assoc-in [:subscriptions id] compute-fn))

;; 事件分发
(defn dispatch
  "分发事件到系统中"
  [event]
  (let [event-id (first event)
        handler (get-in @prompt-space [:event-handlers event-id])]
    (when handler
      (swap! prompt-space handler event))))

;; 订阅访问
(defn subscribe
  "访问订阅数据"
  [query-v]
  (let [query-id (first query-v)
        computation (get-in @prompt-space [:subscriptions query-id])]
    (r/track #(computation @prompt-space query-v))))
