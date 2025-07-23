from motion_merge import motion_merge, api_motion_merge

words = ['모르는단어2','성토','모르는단어']

data = motion_merge(words, send_type='api')


api_motion_merge(*data)
