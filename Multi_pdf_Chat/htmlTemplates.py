css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .icon {
    font-size: 28px;
    color: white;
    margin-right: 10px;
}
.chat-message .message {
    width: 100%;
    color: #fff;
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="icon">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="icon">👤</div>
    <div class="message">{{MSG}}</div>
</div>
'''
