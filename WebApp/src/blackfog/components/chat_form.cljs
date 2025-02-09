(ns blackfog.components.chat-form
  (:require [reagent.core :as r]
            [datascript.core :as d]
            [markdown.core :refer [md->html]]
            [clojure.string :as str]
            [cljs.core.async :refer [go chan <! >!]]
            [blackfog.nexus.core :refer [->NexusPoint]]
            [blackfog.nexus.api :as api]
            [blackfog.nexus.message :as msg]
            [blackfog.nexus.personas-data :as ps-data]
            [blackfog.nexus.personas :as ps]
            [blackfog.components.download :refer [download-button
                                                  upload-button]]
            [blackfog.app-state.core :refer [app-state threads
                                             personas nexus]]
            [blackfog.components.image :refer [ImageUploadControl]]
            [blackfog.components.textarea :refer [TextArea]]
            [blackfog.components.dropdown :refer [DropDown]]
            ))

;; helper
(defn path [& args]  (into [:nexus-port :default] args))

;; components/chat/buttons.cljs
(defn SendButton [{:keys [is-sending on-click]}]
  [:button.button.is-primary.is-small.mr-2
   {:class (when is-sending "is-loading")
    :on-click on-click} "Send"])

(defn RefleshMemoryButton [{:keys [is-sending on-click]}]
  [:button.button.is-primary.is-small.mr-2
   {:class (when is-sending "is-loading")
    :on-click on-click} "Send-Many"])

(defn CleanMemoryButton [app-state nexus]
  (r/with-let [history   (r/cursor app-state [:db :avatar :history])]
    [:button.button.is-small.is-primary
     {:on-click #(msg/reset-history! nexus [])} "清除"]))

(defn ChatControls [{:keys [app-state is-sending
                            chat-history on-linxia-send
                            nexus
                            on-reflesh-memory]}]
  (r/with-let [page-state (r/cursor app-state [:page :index])
               tag        (r/cursor page-state [:active])]
    (let [data-handler identity
          c-tag @tag]
      [:div.level-right
       [:div.level-item [SendButton {:is-sending is-sending
                                     :on-click on-linxia-send}]]
       [:div.level-item [ImageUploadControl {:app-state app-state}]]
       [:div.level-item [RefleshMemoryButton {:is-sending is-sending
                                              :on-click on-reflesh-memory}]]
       [:div.level-item
        [download-button chat-history "chat-history.json" data-handler]]
       [:div.level-item [upload-button (fn [data]
                                         (reset! chat-history data)
                                         (js/alert "记忆已恢复！"))]]
       [:div.level-item [CleanMemoryButton app-state nexus ]]])))


(defn InputControls [{:keys [app-state is-sending
                             chat-history on-send
                             on-reflesh-memory]}]
  [:div.level-left
   [:div.level-item [SendButton {:is-sending is-sending
                                 :on-click on-send}]]])

(def chat-box-styles
  {:chat-container {:position "fixed"
                    :bottom "5px"    ;; 距离底部留出空间
                    :left "5px"     ;; 靠右对齐
                    :right "5px"
                    ;;:width "600px"    ;; 固定宽度
                    :max-width "90vw" ;; 移动端适配
                    :border-radius "6px"
                    :box-shadow "0 4px 20px rgba(0, 0, 0, 0.15)"
                    :background-color "white"
                    :z-index 9999}

   :input-area {:padding "15px"
                :border-top "1px solid #eee"}

   :controls {:margin-bottom "10px"}

   :textarea {:border "1px solid #e0e0e0"
              :border-radius "8px"
              :padding "10px"
              :transition "border-color 0.3s ease"
              :resize "none"}})

;; components/chat/chat_box.cljs
(defn NarrationChatBox [{:keys [app-state nexus page-state history]}]
  (r/with-let [input-value   (r/atom "")
               is-sending    (r/atom false)
               is-extracting (r/atom false)
               img     (r/cursor app-state [:db :avatar :current-image])]
    (let [callback (fn [chunk]
                     (println "==++==")
                     (reset! is-sending false)
                     (reset! input-value ""))
          handle-send (fn [e]
                        (when (not (str/blank? @input-value))
                          (reset! is-sending true)
                          (ps/recive! nexus @input-value callback)))
          handle-reflesh-memory (fn [e])
          handle-key-press  (fn [e]
                              (when (and (not-empty @input-value)
                                         (= (.-key e) "Enter"))
                                (if (not @is-sending)
                                  (handle-send e))))
          handle-linxia-send (fn [e])]
      [:footer.is-flex-align-items-flex-end.mt-auto.is-fixed-bottom
       [:div.box {:style (:chat-container chat-box-styles)}
        [:div.level {:style (:controls chat-box-styles)}
         [ChatControls
          {:app-state app-state
           :img @img
           :nexus nexus
           :is-sending @is-sending
           :chat-history (into [] (map msg/msg->map history))
           :on-linxia-send  handle-linxia-send
           :on-reflesh-memory    handle-reflesh-memory}]]
        [:div.level
         [:div.level-left.is-flex-grow-1
          [:div.level-item.is-flex-grow-1
           [TextArea
            (merge {:value input-value
                    :on-change #(reset! input-value %)
                    :on-key-press handle-send
                    :placeholder "说点什么..."
                    :min-rows 2
                    :max-rows 4}
                   {:style (:textarea chat-box-styles)})]]]
         [InputControls
          {:app-state app-state
           :img @img
           :is-sending @is-sending
           :chat-history history
           :on-send handle-send
           :on-reflesh-memory handle-reflesh-memory}]]]])))


;; components/chat/common.cljs
(defn RoleHeader [{:keys [role] :as item}]
  (let [role-name (case role
                    "user" "User"
                    "assistant" "Assistant"
                    "system" "System"
                    :assistant "Assistant"
                    :user "User"
                    :system "System")
        role-class (name role)]
    [:div.message-header
     [:span {:class role-class} role-name]]))

(defn ActionButton [{:keys [icon label on-click]}]
  [:button.button.is-small.mr-2
   {:on-click on-click}
   (if icon
     [:span [:i {:class icon}]]
     [:span label])])

;; components/chat/narration_edit.cljs
(defn NarrationEdit [{:keys [app-state page-state nexus
                             item id role row-id content
                             on-status-change]}]
  (r/with-let [input-value (r/atom content)]
    (let [handle-save (fn []
                        (msg/update-message nexus id {:content @input-value})
                        (on-status-change :view))]
      [:div.message.box
       [RoleHeader {:role role}]
       [TextArea
        {:value input-value
         :on-change #(reset! input-value %)
         :placeholder "说点什么..."
         :min-rows 6
         :max-rows 20}]
       ;;[TextArea value]
       [:div.control.is-expanded
        [ActionButton {:label "完成"
                       :on-click handle-save}]
        [ActionButton {:label "放弃"
                       :on-click #(on-status-change :view)}]]])))


;; components/chat/narration_view.cljs
(defn NarrationView [{:keys [app-state page-state nexus
                             item id role row-id content
                             on-status-change]}]
  (r/with-let [is-hold        (r/atom false)]
    (let [handle-spin         (fn [e])
          handle-regenerate (fn [e])
          handle-delete (fn [e]
                          (msg/remove-message nexus id))]

      [:div.control.is-expanded
       [:div.message.box {:class (str (name role) "-message")}
        [RoleHeader {:role role}]

        [:div.content.narration-content
         [:div.markdown-body.px-4.py-2
          {:style {:text-indent "2em"  ;; 首行缩进

                   :padding-left "1em" ;; 整体左侧缩进

                   :white-space "pre-wrap"}
           :dangerouslySetInnerHTML
           {:__html (md->html (str/replace content #"(?<!\n)\n(?!\n)" "\n\n"))}}]]

        [:div.level
         [:div.level-left
          [:div.level-item
           [ActionButton {:label "置顶"
                          :on-click handle-spin}]]
          [:div.level-item
           [ActionButton {:icon "fa-solid fa-arrows-rotate"
                          :on-click handle-regenerate}]]
          [:div.level-item
           [ActionButton {:label "修改"
                          :on-click #(on-status-change :edit)}]]
          [ActionButton {:label "解析"
                         :on-click (fn [x] "")}]
          [ActionButton {:icon "fa-regular fa-trash-can"
                         :on-click handle-delete}]]]]])))

;; components/chat/narration_line.cljs
(defn NarrationLine [app-state page-state nexus item row-id]
  (r/with-let [status! (r/atom :view)]
    (let [{:keys [role content id] } item]
      [:div.control
       (let [props {:app-state app-state
                    :page-state page-state
                    :nexus nexus
                    :item item
                    :row-id row-id
                    :id id
                    :role role
                    :content content
                    :on-status-change #(reset! status! %)}]
         (case @status!
           :view [NarrationView props]
           :edit [NarrationEdit props]))])))

;; components/chat/conversation_panel.cljs
(defn HistoryPanel [{:keys [history app-state page-state nexus tag]}]
  [:div.column.is-10
   [:div.is-flex-grow-1
    {:style {:max-height  "80vh"
             :overflow-y  "auto"
             :overflow-x  "hidden"}}
    (for [[row-id item] (map-indexed vector history)]
      ^{:key row-id}
      [NarrationLine app-state page-state nexus item row-id])]])


(defn presence-badge [status]
  (case status
    :online [:span.tag.is-success.is-rounded.ml-2 "在线"]
    ;;:typing [:span.tag.is-warning.is-rounded "输入中..."]
    :offline [:span.tag.is-light.is-rounded.ml-2 "离线"]
    [:span.tag "未知"]))

(defn PresenceActionButton [{:keys [label on-click tag]}]
  (let [status (if (= @tag (keyword label))
                 :online
                 :offline)]
    [:button.button.is-small
     {:on-click on-click
      :style {:position :relative}}
     [:p label  [presence-badge status]]]))


;; components/chat/diary_panel.cljs
(defn DiaryPanel [{:keys [nexus avatar diary app-state tag]}]
  (r/with-let [metadatas       (r/cursor app-state [:db :avatar :history/meta])
               history         (r/cursor app-state [:db :avatar :history])]
    [:div.column
     [:div.content
      {:style {:max-height  "85vh"
               :overflow-y  "auto"
               :overflow-x  "hidden"}}
      [:h3 "化身类型 | " (:figure/name avatar)]
      [:p [:b "对话轮数：" (/ (count @history)  2)]]
      (for [[label key] [["感性" :figure/EmotionalPerception]
                         ["潜意识" :figure/SubconsciousDimension]
                         ["直觉" :figure/IntuitiveThinking]
                         ;; ... 其他属性
                         ]]
        ^{:key key}
        [:div [:b (str label "：")] (get diary key)])
      [:hr]
      [:div.menu
       [:li
        [PresenceActionButton
         {:label "Eve"
          :tag tag
          :on-click #(do
                       (reset! tag :Eve))}]]
       [:li
        [PresenceActionButton
         {:label "LinXia"
          :tag tag
          :on-click #(reset! tag :LinXia)}]]
       [:li
        [PresenceActionButton
         {:label "bonsai"
          :tag tag
          :on-click #(reset! tag :bonsai)}]]
       ]

      [:hr]
      [:p (str (msg/cache nexus))]]]))

;; components/chat/form.cljs
(defn Form [app-state]
  (r/with-let [nexus-port (r/cursor app-state (path))
               page-state (r/cursor app-state [:page :index])
               tag        (r/cursor page-state [:active])
               history    (r/cursor app-state (path :history))
               reciver    (r/cursor app-state (path :personas :personas))]
    (let [nexus (->NexusPoint nexus-port)
          _     (ps-data/init-personas nexus @tag)
          _     (msg/load-msg nexus)
          c-tag @tag]

      [:div.container.hero.is-fullheight.is-fluid.is-clipped
       [:div.columns
        [HistoryPanel {:history @history
                       :nexus nexus
                       :tag @tag
                       :page-state page-state
                       :app-state app-state}]
        [DiaryPanel {:nexus nexus
                     :tag tag
                     :app-state app-state}]]
       [NarrationChatBox {:app-state app-state
                          :nexus nexus
                          :page-state page-state
                          :history @history}]])))
