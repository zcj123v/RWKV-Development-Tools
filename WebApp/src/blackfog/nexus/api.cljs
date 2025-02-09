(ns blackfog.nexus.api
  (:require
   [cljs.core.async :refer [go chan <! >! go-loop put!] :as async]
   [ajax.core :refer [POST]]
   [blackfog.dsl.core :as dsl]
   [blackfog.api :as api]
   [blackfog.nexus.personas-data :as ps-data]
   [blackfog.nexus.message :as msg :refer [->Message]]))

;; call back  protocol
(defprotocol CallBackNexusPoint
  "定义通信节点的基本行为"
  (send! [this msg callback ] "发送消息并处理回调"))

;; steram protocol
(defprotocol StreamingNexusPoint
  "流式通信节点的扩展行为"
  (stream! [this config msg callback] "发送消息并进行流式处理")
  (append-cache! [this content] "追加内容到缓存")
  (flush-cache! [this msg postfix-fn] "将缓存内容提交到历史记录")
  (extract-info [this user-input] "解析用户输入"))

(defn callback-send-handler [this state content callback]
  (let [{:keys [history prefix config]} @state
        message (msg/create-message this "user" content )
        context (into (or prefix []) (conj history message))
        handle-response
        (fn [response]
          (let [response-msg (msg/create-message this "assistant" response)]
            (swap! state merge
                   {:output response
                    :history (conj history message
                                   response-msg)}))
          (callback  response))]
    (go (<! (api/call-api-callback config context handle-response)))))




(defn extrac-info [this state user-input]
  (let []))

(defn stream-send-handler [this state config msg callback]
  (let [context (reduce (fn [m n]
                          (conj m (select-keys n [:role :content])))
                        []  msg)
        postfix     (ps-data/postfix-in-receive this)
        middlewares (ps-data/middlewares-in-receive this)
        handle-stream (fn [chunk]
                        (if (= chunk :done)
                          (do (flush-cache! this msg postfix)
                              (let [output (msg/output this)]
                                (callback output)
                                (doseq [middleware middlewares]
                                  (middleware this state output))))
                          (do (append-cache! this chunk))))]
    (api/call-stream-api config context handle-stream
                         :callback-reasoning
                         #(swap! state update :output/think str %))))

(defn stream-flush-cache [this state msg postfix-fn]
  (let [raw-content (msg/cache this)
        cached-content (try (postfix-fn raw-content)
                            (catch js/Error e
                              (let [] (println "wrong-->" (.-message e))
                                   raw-content)))
        user-input     (-> msg last :content)
        request-msg    (msg/create-message this "user" user-input)
        response-msg   (msg/create-message this "assistant" cached-content)]

    (swap! state
           (fn [current-state]
             (-> current-state
                 (update :history conj request-msg response-msg)
                 (assoc :output raw-content)
                 (assoc :cache ""))))
    (msg/save-msg this)))
