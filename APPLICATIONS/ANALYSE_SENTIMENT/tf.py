# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:49:28 2019

@author: Juliette
"""
import time
import json
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler  
from tweepy.streaming import StreamListener 
import sentiment_module as s

ckey='sroswScAmttwtccdY2x4OzBkx'
csecret='GcOwemcXgSsNBFOoVVxIQCPiE7QyNeY4kfITaapsMeVCXI35uU'
atoken='1100481382525095937-oLydOfEtLnir4hfI0MAUcVR4ROzqLk'
asecret='83NgoJFNC8JBFbbxe8nQMd50eOtRvA8SLca6kX3mqC4vc'

class Listener (StreamListener):
    def on_data(self,data):
        try:
            #print (data) --only for full data
#            tweet = data.split (',"text":"')[1].split('","source')[0] #trims down to text
           
            #saveThis = str(time.time())+ '::'+tweet #avoid common punctuation
#            saveThis=open('tweetDB2.csv','a')
#            saveThis.write(tweet)
#            saveThis.write('\n')
#            saveThis.close()
#            
            '''
            saveFile=open('tweetDB.csv','a')#create a file iot save the data
            saveFile.write(data)
            saveFile.write('\n')
            saveFile.close()
            '''
            all_data = json.loads(data)
            tweet = all_data["text"]
            sentiment_value,confidence = s.sentiment(tweet)
            print (tweet,sentiment_value,confidence)
            
            if confidence*100 >= 80 :
                output = open ("twitter-out.txt","a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()
                
                            
            return True
        except BaseException as e: #rate limitation 
            print ('failed ondata,', str(e))
            time.sleep(8) #avoids constant reconnection ico of rate limitation
    def on_error(self,status):
        print (status)
        
auth =OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)
twitterStream= Stream(auth,Listener())
twitterStream.filter(track=["game of thrones"])         