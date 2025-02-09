(ns blackfog.dsl.styles
  (:require [clojure.string :as str]
            [cljs-time.core :as t]
            [cljs-time.format :as tf]
            [cljs-time.coerce :as tc]))


(defn user [content]
  {:role "user"
   :content (str content)})

(defn assistant [content]
  {:role "assistant"
   :content (str content)})

(defn system [content]
  {:role "system"
   :content (str content)})

(defn page [& coll] (into [] coll))

;;
(defn start-of-today [date]
  (-> date
      tc/to-local-date
      tc/to-date-time))

(defn end-of-today [date]
  (-> date
      tc/to-local-date
      (t/plus (t/days 1))
      (t/minus (t/millis 1))
      tc/to-date-time))

(defn- day-of-week [date]
  (t/day-of-week date))

(defn start-of-week [date]
  (let [current-day (day-of-week date)]
    (t/minus date (t/days (dec current-day)))))

(defn end-of-week [date]
  (let [current-day (day-of-week date)]
    (t/plus date (t/days (- 7 current-day)))))

(defn start-of-month [date]
  (tc/from-date (js/Date. (t/year date) (dec (t/month date)) 1)))

(defn end-of-month [date]
  (let [next-month (t/plus date (t/months 1))]
    (t/minus (start-of-month next-month) (t/days 1))))

(defn start-of-quarter [date]
  (let [month (t/month date)
        quarter-start-month (* (quot (dec month) 3) 3)]
    (tc/from-date (js/Date. (t/year date) quarter-start-month 1))))

(defn end-of-quarter [date]
  (t/minus (start-of-quarter (t/plus date (t/months 3))) (t/days 1)))

(defn days-between [start end]
  (t/in-days (t/interval start end)))
;;

(defn trim-args [args]
  (map (comp str/trim str) args))

(defn p [& args]
  (str/join   (map (comp str/trim str) args)) )

(defn row [& args]
  (str/join "\n"  (map (comp str/trim str) args)) )

(defn rows [& args]
  (str/join "\n\n"  (map (comp str/trim str) args)) )

(defn bold [& args]
  (str "**" (str/trim (str/join args)) "**"))
(def b bold)

;; 斜体
(defn italic [& args]
  (str "*" (apply p args) "*"))


(defn strikethrough [& args]
  (str "~~" (apply p args) "~~"))

(defn h1 [& args] (str "# " (apply p args)))

(defn h2 [& args] (str "## " (apply p args)))

(defn h3 [& args] (str "### " (apply p args)))

(defn h4 [& args] (str "#### " (apply p args)))

;; table
(defn table [headers & rows]
  (let [divider (str/join "|" (repeat (count headers) "---"))
        header-row (str/join "|" headers)]
    (str/join "\n"
              (concat [header-row (str "|" divider "|")]
                      (map #(str/join "|" %) rows)))))


(defn image [url alt-text & [title]]
  (let [title-part (when title (str " \"" title "\""))]
    (str "![" alt-text "]" "(" url title-part ")")))

(defn hr [] "\n---\n")

;;list

(defn listform [type & items]
  (let [prefixes (if (= type :ordered)
                   (map #(str (inc %1) ". ") (range (count items)))
                   (repeat "- "))]
    (str/join "\n" (map #(str %1 %2) prefixes items))))

(defn listitem [& args]
  (str " - " (apply p args)))

(defn task-list-item [checked? & args]
  (str "- [" (if checked? "x" " ") "] " (apply p args)))


;;  code block
(defn GATE [role status-bar]
  (str "<|GATE|> " "**" role "**: " status-bar))

(defn inner-thought [& args]
  (str "\n> 💭 " (str/join (map (comp str/trim str) args)) "\n"))

;; 代码块
(defn code-inline [& args]
  (str "`" (str/join " " (map (comp str/trim str) args))  "`"))

(defn code [code args]
  (str "\n```" code "\n"
       (str args)
       "\n```\n"))

(defn block [args]
  (str "\n```\n"
       (str args)
       "\n```\n"))


(defn format-edn [obj]
  (str "\n```edn\n\n"
       (with-out-str
         (cljs.pprint/pprint obj))
       "\n```\n"))

;; 时间函数
(def custom-formatter (tf/formatter "yyyy-MM-dd HH:mm:ss"))

(defn timestamp []
  (str (tf/unparse custom-formatter (t/now))))



;; 新增的三个时间戳函数
(defn yesterday-timestamp []
  (str (tf/unparse custom-formatter (t/yesterday))))

(defn tomorrow-timestamp []
  (str (tf/unparse custom-formatter (t/plus (t/now) (t/days 1)))))


(defn this-week-timestamp []
  (let [now (t/now)
        start-of-week (start-of-week now)
        end-of-week (end-of-week now)]
    [(str (tf/unparse custom-formatter start-of-week))
     (str (tf/unparse custom-formatter end-of-week))]))

(defn this-month-timestamp []
  (let [now (t/now)
        start-of-month (start-of-month now)
        end-of-month (end-of-month now)]
    [(str (tf/unparse custom-formatter start-of-month))
     (str (tf/unparse custom-formatter end-of-month))]))


(defn status [type args]
  (let [icon (case type
               :success "✅"
               :warning "⚠️"
               :error   "❌"
               :info    "ℹ️"
               "")]
    (str icon " " (str/join "\n" (map (comp str/trim str) args)))))

(defn card [title & content]
  (str "### " title "\n"
       "---\n"
       (str/join "\n" (map (comp str/trim str) content))
       "\n---\n"))

(defn badge [label value & [color]] ; 修正参数名拼写错误
  (let [color (or color "blue")]
    (str "![" label "](https://img.shields.io/badge/"
         (str/replace label " " "_") "-"
         (str/replace value " " "_") "-"
         color ")")))


(defn alert [type & content]
  (let [icon (case type
               :tip    "💡"
               :note   "📝"
               :warn   "⚠️"
               :danger "🚨"
               :bug    "🐛"
               :rocket "🚀"
               "ℹ️")]
    (str icon " **" (str/upper-case (name type)) ":** "
         (str/join " " content))))


(defn details [summary & content]
  (str "<details>\n<summary>" summary "</summary>\n\n"
       (apply p content)
       "\n</details>"))


(defn link [& links]
  (when (odd? (count links))
    (throw (js/Error. "link function requires even number of arguments")))
  (str/join "\n" (for [[text url] (partition 2 links)]
                   (str "- [" text "](" url ")"))))

(defn color [hex & args]
  (str "<span style=\"color:" hex "\">" (apply p args) "</span>"))

(defn timeline [& events]
  (str/join "\n" (for [[time desc] (partition 2 events)]
                   (str "- **" time "**: " desc))))
