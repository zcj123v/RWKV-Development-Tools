(ns blackfog.api
  (:require [cljs.core.async :refer [go chan <! >! go-loop] :as async]
            [cljs-http.client :as http]
            [reagent.core :as r]
            [datascript.core :as d]
            [clojure.string :as str]
            [clojure.spec.alpha :as s]
            [blackfog.persets.services :refer [services]]))

;; model
(def models
  {:rwkv          :rwkv
   :gemini/flash.think "gemini-2.0-flash-thinking-exp"
   ;;:claude/sonnet "claude-3-5-sonnet-20240620"
   :claude/sonnet "claude-3-5-sonnet-20241022"
   :claude/opus "claude-3-opus-20240229"
   :gemini/pro "gemini-1.5-pro"
   :gemini/pro.1206 "gemini-exp-1206"
   :gemini/flash.v2 "gemini-2.0-flash-exp"
   :gemini/flash "gemini-1.5-flash-latest"
   :llama   "llama-3.1-405b-instruct"
   :chatgpt/mini-o1 "o1-mini-2024-09-12"
   :gpt4/mini  "gpt-4o-mini"
   :gpt4  "gpt-4"
   :mixtral "mixtral-8x7b-instruct"
   :deepseek "deepseek-chat"
   :deepseek/origin "deepseek-reasoner"
   :deepseek/r1 "deepseek-r1"})


(defn create-request-params [config msg]
  "Extracts and validates request parameters"
  (let [{:keys [api/model model/temperature model/top_p
                model/repetition_penalty
                model/presence_penalty model/frequency_penalty]} config]
    (cond-> {:model model
             :messages msg}
      temperature        (assoc :temperature temperature)
      repetition_penalty (assoc :repetition_penalty repetition_penalty)
      top_p              (assoc :top_p top_p)
      presence_penalty  (assoc :presence_penalty presence_penalty)
      frequency_penalty (assoc :frequency_penalty frequency_penalty))))

(defn create-headers [config]
  {"Authorization" (str "Bearer " (:api/sk config))
   "Content-Type" "application/json;charset=\"utf-8\""})

(defn call-api [config msg]
  (let [services-selected (get config :base/services :default)
        config (merge (get services services-selected) config)
        url  (str (:api/url config) "/chat/completions")
        params    (create-request-params config msg)
        extractor (get config :fn/extractor identity)
        validator (get config :fn/validator identity)
        max-retries (get config :api/max-retries 2)
        headers   (create-headers config)
        error-ch  (chan)]
    (go-loop [retry-count 0]
      (if (>= retry-count max-retries)
        {:error :max-retries-exceeded
         :retry-count retry-count
         :last-attempt-time (js/Date.now)
         :config (dissoc config :api/sk)} ;; 添加更多上下文，但排除敏感信息
        (let [{:keys [success body error] :as response}
              (<! (http/post url {:with-credentials? false
                                  :headers headers
                                  :json-params params}))]
          (cond
            error         {:error error}
            (not success) (recur (inc retry-count))
            :else (let [extracted (-> body :choices first :message :content extractor)]
                    (if (validator extracted)
                      extracted
                      (recur (inc retry-count))))))))))

;;================================


(defn handle-stream-chunk [chunk]
  (let [lines (-> chunk (str/split #"data: "))]
    (keep (fn [line]
            (when (and (not (str/blank? line))
                       (not= line "[DONE]"))
              (try
                (let [parsed (-> line
                                 (js/JSON.parse)
                                 (js->clj :keywordize-keys true))
                      content (get-in parsed [:choices 0 :delta :content])
                      reasoning_content (get-in parsed [:choices 0 :delta :reasoning_content])
                      finish-reason (get-in parsed [:choices 0 :finish_reason])]
                  (when (or content reasoning_content finish-reason)
                    {:content content
                     :reasoning_content reasoning_content
                     :finished? (= finish-reason "stop")}))
                (catch js/Error _)))) lines)))

(defn stream-api [config msg callback & {:keys [callback-reasoning]}]
  (let [config (merge (:default services) config)
        url (str (:api/url config) "/chat/completions")
        params (-> (create-request-params config msg)
                   (assoc :stream true))
        headers (create-headers config)
        xhr (js/XMLHttpRequest.)
        is-done? (atom false)]
    (.open xhr "POST" url)
    (doseq [[k v] headers]
      (.setRequestHeader xhr k v))

    ;; 添加一个处理数据的辅助函数
    (letfn [(process-new-data []
              (let [new-data (subs (.-responseText xhr)
                                   (or (.-lastIndex xhr) 0))]
                (set! (.-lastIndex xhr) (count (.-responseText xhr)))
                (doseq [{:keys [content finished?
                                reasoning_content]} (handle-stream-chunk new-data)]
                  (when (and callback-reasoning reasoning_content)
                    (callback-reasoning reasoning_content))
                  (when content
                    (callback content))
                  (when finished?
                    (reset! is-done? true)
                    (callback :done)))))]

      (set! (.-onreadystatechange xhr)
            (fn []
              (case (.-readyState xhr)
                3 (process-new-data)  ;; 数据正在到达
                4 (do                 ;; 请求完成
                    (process-new-data)
                    (if (not @is-done?)
                      (callback :done)))
                nil))))

    (.send xhr (js/JSON.stringify (clj->js params)))))

(def call-stream-api stream-api)

;; 原有的 call-api 函数保持不变，用于非流式调用

(defn call-api-callback [config msg callback]
  (let [ch (call-api config msg)]
    (go (callback (<! ch)))))

;; Promise-based version for JavaScript interop
(defn call-api-promise [config msg]
  (js/Promise. (fn [resolve reject]
                 (call-api-callback config msg #(if (:error %)
                                                  (reject (:error %))
                                                  (resolve %))))))
;; ================================
