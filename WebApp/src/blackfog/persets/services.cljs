(ns blackfog.persets.services)


(def services
  {:default  {:api/url   "https://yunwu.ai/v1"
              :api/max-retries 1
              :api/retry-delay 1000
              :api/model "gemini-1.5-pro"
              :api/sk ""
              :model/temperature 1.4
              :model/top_p 0.6
              :model/repetition_penalty 1.0
              :model/presence_penalty 0.1
              :model/frequency_penalty 1.0
              :fn/extractor   identity
              :fn/validtor    identity}
   :claude   {:api/url   "https://yunwu.ai/v1"
              :api/max-retries 1
              :api/retry-delay 1000
              :api/model "claude-3-5-sonnet-latest"
              :api/sk ""
              :model/temperature 1.8
              :model/top_p 0.7
              :model/repetition_penalty 1.0
              :model/presence_penalty 0.1
              :model/frequency_penalty 1.0
              :fn/extractor   identity
              :fn/validtor    identity}

   :deepseek/origin {:api/url   "https://api.deepseek.com"
                     :api/max-retries 1
                     :api/retry-delay 1000
                     :api/model "deepseek-reasoner"
                     :api/sk "s"
                     :model/temperature 0.65
                     :model/top_p 0.65
                     :fn/extractor   identity
                     :fn/validtor    identity}

   :deepseek {:api/url   "https://yunwu.ai/v1"
              :model/temperature 0.65
              :model/top_p 0.65
              :api/max-retries 1
              :api/retry-delay 1000
              :api/model "deepseek-r1"
              :api/sk ""
              :fn/extractor   identity
              :fn/validtor    identity}

   :extractor {:api/url   "https://yunwu.ai/v1"
               :api/max-retries 1
               :api/retry-delay 1000
               :api/model "gemini-1.5-pro"
               :api/sk ""
               :model/temperature 1.0
               :model/top_p 0.65
               :model/presence_penalty 0
               :model/frequency_penalty 0
               :model/repetition_penalty 1.0
               :fn/extractor   identity
               :fn/validtor    identity}}
  )
