<template>
<div>
    <basic-vue-chat :title="'SOLOIST Interact Interface'" :initial-feed="feed" @newOwnMessage="test" :new-message="message" ref="child"/>
      <b-card class="mt-3" header="Tracker Memory" style="text-align:left">
      <pre class="m-0">{{ memory }}</pre>
    </b-card>
</div>
</template>

<style lang="scss">
$primary : red;
$font-weight : 800 ;
@import "@/assets/scss/main.scss";
$primary : red;
$font-weight : 800 ;
</style>


<script>
// import { EventBus } from "./event-bus.js";
import moment from 'moment'
import BasicVueChat from 'basic-vue-chat'
import Vue from 'vue'
import axios from 'axios'
import VueAxios from 'vue-axios'

// var VueCookie = require('vue-cookie');
// Tell Vue to use the plugin
// Vue.use(VueCookie);
// // import qs from 'qs';
Vue.use(VueAxios, axios)

const cfeed = [
  
  {
    id: 1,
    author: 'System',
    imageUrl: 'http://path/to/image',
    contents: 'Welcome to our system. Please chat with the agent in the below window.',
    date: ''
  },
]


export default {
  name: 'chat',
  components: {
    BasicVueChat
  },
  data: function () {
    return {
      message: {
            id: 0,
            author: 'Person',
            imageUrl: 'http://path/to/image',
            image: '',
            contents: 'hi there',
            date: '16:30'
      },
      feed : cfeed,
      all_data : [],
      memory : {}
    }
  },
  methods : {
      test (message, image, imageUrl) {
        //   alert(message)
      const newOwnMessage = {
        id: 1,
        author : 'Person',
        contents: message,
        image: image,
        imageUrl: imageUrl,
        date: moment().format('HH')
      }
      newOwnMessage
      this.all_data.push(message)
      this.$emit('newOwnMessage', message)
    
    },
  },
mounted() {
      this.$on("newOwnMessage", (p) => {
        p
        axios({
              method: 'POST',
              url: 'http://104.210.220.116:8081/generate',
              data: {'msg':this.all_data},
              
          }).then(response => {

              this.all_data.push(response.data.response)

                const newOwnMessage = {
                    id: 1,
                    author : 'Agent',
                    contents: response.data.response,
                    image: null,
                    imageUrl: null,
                    date: moment().format('HH:MM:SS')
                }
              this.memory = response.data.memory
              // alert(this.memory)
              this.message = newOwnMessage
              if (response.data.followup != '')
              {
                const newOwnMessage = {
                    id: 1,
                    author : 'Agent',
                    contents: response.data.followup,
                    image: null,
                    imageUrl: null,
                    date: moment().format('HH:MM:SS')
                }
                this.$refs.child.pushToFeed(newOwnMessage)
              }
              // this.$emit('followup', 't')
              console.log(response);
          }).catch(function (error) {
              console.log(error);
          });
        })
    }
}
</script>

