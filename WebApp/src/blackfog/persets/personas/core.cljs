(ns blackfog.persets.personas.core
  (:require
   [blackfog.dsl-protocol.impl]
   [blackfog.persets.personas.private.eve :as Eve]
   [blackfog.persets.personas.private.bonsai :as bonsai]
   [blackfog.persets.personas.private.bonsai-sonnet :as bonsai-sonnet]
   [blackfog.persets.personas.private.linxia :as linxia]
   [blackfog.persets.personas.private.linxia-first :as linxia-first]

   [blackfog.persets.extractor :as extractor]
   [blackfog.nexus.message :as msg]
   [blackfog.persets.personas.the-reader :refer [TheReaderPrefix]]))



(def personas-preset
  {:Eve
   {:personas/name  :Eve
    :prefix/reader TheReaderPrefix
    :service/config :claude
    :recive               Eve/Recive
    :recive.fn/postfix    extractor/text->pure-text
    :recive.fn/middlewares [(fn [this state text]
                              (swap! state assoc :output.iphone/edn
                                     (extractor/iphone->inner-edn text)))]
    :speak  nil
    :snap/perceptual  Eve/PerceptualThinking
    :snap/rational    Eve/RationalThinking}

   :bonsai/dave
   {:personas/name  :bonsai
    :prefix/reader  bonsai-sonnet/TheReaderPrefix
    :service/config :claude
    :recive               bonsai/Recive
    :recive.fn/postfix    (fn [x] x)
    :recive.fn/middlewares [(fn [this state text]
                              (msg/last-message-with-meta
                               this {:output/think (extractor/think->inner text)}))
                            (fn [this state text]
                              (println "====" (msg/last-message this)))]
    :speak  nil}

   :bonsai
   {:personas/name  :bonsai
    :prefix/reader  bonsai-sonnet/TheReaderPrefix
    :service/config :deepseek
    :recive               bonsai/Recive
    :recive.fn/postfix    extractor/think->outer
    :recive.fn/middlewares [(fn [this state text]
                              (msg/last-message-with-meta
                               this {:output/think (extractor/think->inner text)}))
                            (fn [this state text]
                              (println "====" (msg/last-message this)))]
    :speak  nil}


   :LinXia
   {:personas/name :LinXia
    :prefix/reader linxia-first/TheReaderPrefix
    :service/config :deepseek
    :recive  linxia-first/Recive
    :recive.fn/postfix    extractor/think->outer
    :recive.fn/middlewares
    [(fn [this state text]
       (if-let [res (get @state :output/think)]
         (do (msg/last-message-with-meta
              this
              {:output/think res})
             (swap! state assoc :output/think nil))
         (msg/last-message-with-meta
          this
          {:output/think (extractor/think->inner text)})))
     (fn [this state text]
       (msg/last-message-with-meta
        this
        {:output/state (extractor/codeblock-edn text)}))
     (fn [this state text]
       (msg/last-message-with-meta
        this
        {:xml (extractor/xml->map text)}))
     (fn [this state text]
       (println "====" (dissoc (msg/last-message this)
                               :content)))]
    :speak  :task.eve.speak
    :snap   :task.eve.snap}

   :LinXia/oringin
   {:personas/name :LinXia
    :prefix/reader TheReaderPrefix
    :service/config :deepseek
    :recive  linxia/Recive
    :recive.fn/postfix    extractor/think->outer
    :recive.fn/middlewares
    [(fn [this state text]
       (if-let [res (get @state :output/think)]
         (do (msg/last-message-with-meta
              this
              {:output/think res})
             (swap! state assoc :output/think nil))
         (msg/last-message-with-meta
          this
          {:output/think (extractor/think->inner text)})))
     (fn [this state text]
       (msg/last-message-with-meta
        this
        {:output/state (extractor/codeblock-edn text)}))
     (fn [this state text]
       (msg/last-message-with-meta
        this
        {:output/iphone (extractor/iphone->inner-edn text)}))
     (fn [this state text]
       (println "====" (dissoc (msg/last-message this)
                               :content)))]
    :speak  :task.eve.speak
    :snap   :task.eve.snap}



   :Echo
   {:personas/name :Echo
    :service/config :deepseek
    :recive :task.linxia.recive
    :speak  :task.eve.speak
    :snap   :task.eve.snap}
   })
