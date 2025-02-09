(ns blackfog.components.textarea
  (:require [reagent.core :as r]))

;; components/common/text_area.cljs
(defn TextArea [{:keys [value
                        on-change
                        on-key-press
                        placeholder
                        auto-focus?
                        disabled?
                        max-length
                        min-rows
                        max-rows]
                 :or {placeholder "输入消息..."
                      auto-focus? true
                      disabled? false
                      max-length 20000
                      min-rows 2
                      max-rows 8}}]
  (r/with-let [node-ref (atom nil)
               ;; 自动调整高度
               adjust-height! (fn [node]
                                (when node
                                  (set! (.-height (.-style node)) "auto")
                                  (let [scroll-height (.-scrollHeight node)
                                        max-height (* max-rows 20)] ;; 假设每行20px
                                    (set! (.-height (.-style node))
                                          (str (min scroll-height max-height) "px")))))

               ;; 处理输入变化
               handle-change (fn [e]
                               (let [new-value (.. e -target -value)]
                                 (when (and on-change
                                            (<= (count new-value) max-length))
                                   (on-change new-value))
                                 (adjust-height! @node-ref)
                                 (when-let [textarea @node-ref]
                                   (set! (.-scrollTop textarea) (.-scrollHeight textarea)))))

               ;; 处理特殊按键
               handle-key-press (fn [e]
                                  (when (and on-key-press
                                             (not (.-shiftKey e))
                                             (= (.-key e) "Enter"))
                                    (.preventDefault e)
                                    (on-key-press e)))]

    [:textarea.textarea
     {:ref #(when %
              (reset! node-ref %)
              (when auto-focus? (.focus %)))
      :value @value
      :placeholder placeholder
      :disabled disabled?
      :class "has-fixed-size"
      :style {:resize "none"
              :min-height (str (* min-rows 20) "px")
              :max-height (str (* max-rows 20) "px")
              :overflow-y "auto"}
      :on-change handle-change
      :on-key-press handle-key-press}]))
