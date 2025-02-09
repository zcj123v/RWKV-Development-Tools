(ns blackfog.persets.personas.assistant
  (:require
   [clojure.string :as str]
   [reagent.core :as r]
   [blackfog.dsl-protocol.core :refer [prompt-register!]]
   [blackfog.dsl.styles :as style :refer [row rows p b h1 h2 h3
                                          inner-thought block
                                          user assistant system
                                          page GATE hr timestamp
                                          listform listitem]]
   [blackfog.dsl-protocol.impl :refer-macros [prompt defprompt defhc]]))



;; 定义格式化prompt 可以直接用html语法更新
;; 例如[:row ""]
;; 运行结果为{:role "system" :content "琉璃是一个友善的小助手"}
;; 用这种格式的原因是，llm对格式极端敏感，必须移除所有潜在因为换行和不可视字符导致的可能性。
(defprompt System-prompt
  [:system [:p "琉璃是一个友善的小助手"]])


;; 组件化响应式prompt,支持动态传入参数
;; 结果是形成一条 user message.
(defn User-Prompt [turn content]
  (prompt [:user
           [:rows
            [:row [:p "【From】 JohnSmith (约翰·史密斯)\n"]
             [:p "【Time】 " [:timestamp] "\n"]
             [:p "【Turn】 " turn "\n"]
             [:p "【at】 Then physic world"]
             [:hr]]
            [:p content]]]))


;; 风格3,绑定一个app-state,影响全局渲染，高阶用法，初学者不要用
(defonce state (r/atom {}))

;; 结果，形成一个注册在中的 prompt template，用于挂载在某个可视化控件下
(defhc state :Assistant-prompt
  [:assistant [:p "企鹅不会飞"]])


;; 定义标准接受函数，表示智能体接受的数
;; msg 为历史数据
;; user-input 为接受数据
(defn Recive [msg user-input]
  (let [query (prompt [:page System-prompt
                       (User-Prompt 0 "企鹅会不会飞？")
                       (get @state :Assistant-prompt)])
        query (into query msg)
        query (into query [(prompt [:user user-input])])]
    (println "===" query)
    query))
