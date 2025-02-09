(ns blackfog.persets.personas.base
  (:require
   [clojure.string :as str]
   [blackfog.dsl-protocol.core :refer [prompt-register!]]
   [blackfog.dsl.styles :as style :refer [row rows p b h1 h2 h3
                                          inner-thought block
                                          user assistant system
                                          page GATE
                                          listform listitem]]
   [blackfog.dsl-protocol.impl :refer-macros [prompt defprompt defhc]]))
