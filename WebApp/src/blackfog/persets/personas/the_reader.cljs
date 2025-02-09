(ns blackfog.persets.personas.the-reader
  (:require
   [clojure.string :as str]
   [blackfog.dsl-protocol.core :refer [prompt-register!]]
   [blackfog.dsl.styles :as style :refer [row rows p b h1 h2 h3
                                          inner-thought block
                                          user assistant system
                                          page GATE
                                          listform listitem
                                          timestamp]]
   [blackfog.dsl-protocol.impl :refer-macros [prompt defprompt defhc]]))


(defn TheReaderPrefix [turn content]
  (str "【From】 JohnSmith (约翰·史密斯)\n"
       "【Time】 " (timestamp) "\n"
       "【Turn】 " turn "\n"
       "【at】 Then physic world"
       "\n---\n"
       content))
