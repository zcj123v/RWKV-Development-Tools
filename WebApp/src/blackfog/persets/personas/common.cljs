(ns blackfog.persets.personas.common
  (:require
   [blackfog.dsl-protocol.impl]
   [blackfog.persets.personas.assistant :as assistant]
   [blackfog.persets.extractor :as extractor]
   [blackfog.nexus.message :as msg]
   [blackfog.persets.personas.the-reader :refer [TheReaderPrefix]]))

(def personas-preset
  {:Eve
   {:personas/name  :Eve
    :prefix/reader TheReaderPrefix
    :service/config :claude
    :recive               assistant/Recive
    :recive.fn/postfix    extractor/text->pure-text
    :recive.fn/middlewares []
    :speak  nil
    :snap/perceptual  nil
    :snap/rational    nil}
   })
