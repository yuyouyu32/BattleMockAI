import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.env.act_obs_space import *
from rl.env.agent.config import *
from rl.common.logger import logger

class CustomModel(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, hidden_size=256, cell_size=64, mode='rl_train'):
        super(CustomModel, self).__init__()

        self.num_outputs = num_outputs
        self.mode = mode

        self.cell_size = 256
        self.mp_hidden_size = 64
        self.unit_hidden_size = 256
        self.buff_hidden_size = 64
        self.attention_hidden_size = 256
        self.skill_base_embedding_size = 128
        self.skill_type_embedding_size = 64
        self.hero_embedding_size = 64

        self.eps = 1e-7
        #init
        self.initializer = nn.init.kaiming_normal_
        self.activator = F.gelu
        skill_onehot = torch.eye(TotalSkillNum)
        hero_onehot = torch.eye(HeroNum)
        """ buff process """
        self.buff_fc = nn.Linear(BUFF_FEAT_NUM, self.buff_hidden_size)
        """ mp process """
        self.hero_emb = nn.Embedding(HeroNum, self.hero_embedding_size)
        self.mp_fc = nn.Linear(MP_FEAT_NUM + self.hero_embedding_size + self.buff_hidden_size, self.unit_hidden_size)
        """ ally process """
        self.ally_fc = nn.Linear(ALLY_FEAT_NUM + self.buff_hidden_size, self.unit_hidden_size)
        """ enemy process """
        self.enemy_fc = nn.Linear(ENEMY_FEAT_NUM + self.buff_hidden_size, self.unit_hidden_size)
        """ task 1: skill select """
        self.fc = nn.Linear(self.unit_hidden_size * 3, self.cell_size)
        self.skill_dense = nn.Linear(self.cell_size, ACTION_NUM)
        """ task 2: skill target """
        # _skill_target_op
        self.skill_emb = nn.Embedding(TotalSkillNum, self.skill_base_embedding_size)
        self.skill_base_dense = nn.Linear(self.skill_base_embedding_size, self.skill_base_embedding_size)
        self.skill_info_dense = nn.Linear(SKILL_FEAT_NUM, self.skill_type_embedding_size)
        
        # _attention_op
        self.query_merge_fc = nn.Linear(self.unit_hidden_size * 3 + self.skill_base_embedding_size + self.skill_type_embedding_size, self.attention_hidden_size)
        self.query_fc = nn.Linear(self.attention_hidden_size, self.attention_hidden_size)
        self.key_fc = nn.Linear(self.unit_hidden_size, self.attention_hidden_size)
        self.value_fc = nn.Linear(self.unit_hidden_size, self.attention_hidden_size)

        """ output """
        self.value_dense = nn.Linear(self.cell_size, 1)


    def _attention_op(self, unit_num, sub_name, concat_hidden, skill_embedding, unit_features):
        # Convert each TensorFlow operation to its equivalent in PyTorch
        # Return the required outputs
        
        # Create query
        # (bs, len, c_hs) + (bs, len, es) - > (bs, len, a_hs)
        concat_hidden2 = torch.cat([concat_hidden, skill_embedding], 2)
        merge_info_hidden = self.activator(self.query_merge_fc(concat_hidden2))  # Assuming you have created a linear layer named query_merge_fc in the __init__ method
        query = self.query_fc(merge_info_hidden)  # Assuming you have created a linear layer named query_fc in the __init__ method

        # FC
        # (bs, len, unit_num, hs) -> (bs, len, unit_num, a_hs)
        key = unit_features
        key = self.key_fc(key)  # Assuming you have created a linear layer named key_fc in the __init__ method

        value = unit_features
        value = self.value_fc(value)  # Assuming you have created a linear layer named value_fc in the __init__ method

        # (bs, len, a_hs) -> (bs, len, 1, a_hs)
        query = query.unsqueeze(2)
        # Query * key prob
        z = self.attention_hidden_size
        # (bs, len, unit_num, a_hs)
        query_key_dot_elmentwise = query * key
        # (bs, len, unit_num, a_hs) -> (bs, len, unit_num)
        query_key_dot = torch.sum(query_key_dot_elmentwise, dim=-1) / torch.sqrt(torch.tensor(z))
        query_key_score = F.softmax(query_key_dot, dim=-1)
        
        return query_key_score, query_key_dot
    

    def _skill_target_op(self, skill_slot, hero_id, skill_feature, mp_hidden, ally_hidden, enemy_hidden, ally_mask, enemy_mask,
                         skill_t_ally_masks, skill_t_enemy_masks, concat_hidden):
            # Convert each TensorFlow operation to its equivalent in PyTorch
            # Return the required outputs
            
            # Absolute id: (bs, len) + (bs, len) = (bs, len)
            skill_id = skill_slot + 3 * hero_id

            # (N_skill, es) lookup (bs, len)-> (bs, len, es)
            skill_base_embedding = self.skill_emb(skill_id)
            skill_base_embedding = self.activator(self.skill_base_dense(skill_base_embedding)) 
            # ((bs, len, 3, skill_feat_num) gather (bs, len)-> (bs, len, skill_feat_num)
            skill_slot_expanded = skill_slot.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, -1, SKILL_FEAT_NUM)      
            skill_info = torch.gather(skill_feature, 2, skill_slot_expanded).squeeze(2)
            # ((bs, len, type_size) -> (bs, len, hs)
            skill_info_embedding = self.activator(self.skill_info_dense(skill_info)) 
            skill_embedding = torch.cat([skill_base_embedding, skill_info_embedding], 2)

            # Skill ally/enemy mask
            # (bs, len, 3) gather (bs, len) -> (bs, len, 1)
            skill_t_ally_mask = torch.gather(skill_t_ally_masks, dim=-1, index=skill_slot.unsqueeze(-1))
            skill_t_enemy_mask = torch.gather(skill_t_enemy_masks, dim=-1, index=skill_slot.unsqueeze(-1))

            # Attention
            # (bs, len, ally_num, hs) -> (bs, len, ally_num+1, hs)
            mp_1_hidden = mp_hidden.unsqueeze(2)
            mp_ally_hidden = torch.cat([mp_1_hidden, ally_hidden], -2)
            
            # Assuming you have implemented the _attention_op method that returns attention logits and _ in PyTorch
            ally_selected_prob, _ = self._attention_op(ALLY_NUM+1, "ally", concat_hidden, skill_embedding, mp_ally_hidden)
            enemy_selected_prob, _ = self._attention_op(ENEMY_NUM, "enemy", concat_hidden, skill_embedding, enemy_hidden)

            # Skill target valid mask
            # (bs, len, ally_num) * (bs, len, ally_num)-> (bs, len, ally_num)
            ally_selected_prob = ally_selected_prob * ally_mask
            enemy_selected_prob = enemy_selected_prob * enemy_mask 

            # Skill target type mask
            # (bs, len, ally_num) mul (bs, len, 1) -> (bs, len, ally_num)
            # ally_selected_prob = ally_selected_prob * skill_t_ally_mask + self.eps
            # enemy_selected_prob = enemy_selected_prob * skill_t_enemy_mask + self.eps
            ally_selected_logist = torch.exp(ally_selected_prob)
            enemy_selected_logist = torch.exp(enemy_selected_prob)

            return ally_selected_logist, enemy_selected_logist, skill_t_ally_mask, skill_t_enemy_mask   
    

    def model(self, x):
        # Define input layers
        hero_id, mp_feature, mp_buff, ally_feature, ally_buff, enemy_feature, enemy_buff, \
        skill_feature, ally_mask, enemy_mask, skill_mask, skill_t_ally_masks, skill_t_enemy_masks, skill_slot_label = \
             torch.split(x, \
             [1, MP_FEAT_NUM, BUFF_NUM*BUFF_FEAT_NUM, ALLY_NUM*ALLY_FEAT_NUM, ALLY_NUM*BUFF_NUM*BUFF_FEAT_NUM, ENEMY_NUM*ENEMY_FEAT_NUM, ENEMY_NUM*BUFF_NUM*BUFF_FEAT_NUM, \
             ACTION_NUM*SKILL_FEAT_NUM, ALLY_NUM+1, ENEMY_NUM, ACTION_NUM, ACTION_NUM, ACTION_NUM, 1], 2)

        # (bs, len, 1) -> (bs, len) 
        hero_id = hero_id.squeeze(-1).long()
        skill_slot_label = skill_slot_label.squeeze(-1).long()

        """ buff process """
        # (bs, len, buff_num*fnum) -> (bs, len, buff_num, fnum) 
        mp_buff = torch.stack(torch.split(mp_buff, [BUFF_FEAT_NUM for _ in range(BUFF_NUM)], -1), 2)
        #mp_buff_hidden = self.activator(self.buff_fc(mp_buff))
        mp_buff_hidden = self.buff_fc(mp_buff)
        mp_buff_hidden = self.activator(mp_buff_hidden)
        mp_buff_maxpool, _ = torch.max(mp_buff_hidden, 2) # (bs, len, hs)

        # (bs, len, ally_num*buff_num*fnum) -> (bs, len, ally_num, buff_num*fnum) 
        ally_buff = torch.stack(torch.split(ally_buff, [BUFF_NUM*BUFF_FEAT_NUM for _ in range(ALLY_NUM)], -1), 2)
        # (bs, len, ally_num, buff_num*fnum) -> (bs, len, ally_num, buff_num, fnum) 
        ally_buff = torch.stack(torch.split(ally_buff, [BUFF_FEAT_NUM for _ in range(BUFF_NUM)], -1), 3)
        ally_buff_hidden = self.activator(self.buff_fc(ally_buff))
        ally_buff_maxpool, _ = torch.max(ally_buff_hidden, 3) # (bs, len, ally_num, hs)

        # (bs, len, enemy_num*buff_num*fnum) -> (bs, len, enemy_num, buff_num*fnum) 
        enemy_buff = torch.stack(torch.split(enemy_buff, [BUFF_NUM*BUFF_FEAT_NUM for _ in range(ENEMY_NUM)], -1), 2)
        # (bs, len, enemy_num, buff_num*fnum) -> (bs, len, enemy_num, buff_num, fnum) 
        enemy_buff = torch.stack(torch.split(enemy_buff, [BUFF_FEAT_NUM for _ in range(BUFF_NUM)], -1), 3)
        enemy_buff_hidden = self.activator(self.buff_fc(enemy_buff))
        enemy_buff_maxpool, _ = torch.max(enemy_buff_hidden, 3) # (bs, len, enemy_num, hs)

        """ mp process """
        # merge hero embedding to mp_feature
        # (N_hero, es) lookup (bs, len)-> (bs, len, es)
        hero_embedding = self.hero_emb(hero_id)
        # (bs, len, mp_f_num) concat (bs, len, es) concat (bs, len, hs) -> (bs, len, mp_fnum+hero_es+buff_hs)
        mp_feature_c = torch.cat([mp_feature, hero_embedding, mp_buff_maxpool], 2)

        mp_hidden = self.activator(self.mp_fc(mp_feature_c))
        mp_hidden = nn.LayerNorm(mp_hidden.size()[1:], elementwise_affine=False)(mp_hidden)
        debug = mp_hidden

        """ ally process """
        # (bs, len, ally_num*fnum) -> (bs, len, ally_num, fnum) 
        ally_feature = torch.stack(torch.split(ally_feature, [ALLY_FEAT_NUM for _ in range(ALLY_NUM)], -1), 2)
        # (bs, len, ally_num, fnum) concat (bs, len, ally_num, hs) -> (bs, len, ally_num, fnum+hs)
        ally_feature = torch.cat([ally_feature, ally_buff_maxpool], 3)

        # (bs, len, ally_num, all_feat_num) -> (bs, len, ally_num, hs)
        ally_hidden = self.activator(self.ally_fc(ally_feature))
        ally_hidden = nn.LayerNorm(ally_hidden.size()[2:], elementwise_affine=False)(ally_hidden)
        # (bs, len, ally_num, hs) -> (bs, len, hs)
        # filter invalid ally info by learned fc and maxpooling, whether explicit mask is needed ?
        ally_hidden_maxpool, _ = torch.max(ally_hidden, 2)

        """ enemy process """
        # (bs, len, enemy_num*fnum) -> (bs, len, enemy_num, fnum) 
        enemy_feature = torch.stack(torch.split(enemy_feature, [ENEMY_FEAT_NUM for _ in range(ENEMY_NUM)], -1), 2)
        # (bs, len, enemy_num, fnum) concat (bs, len, enemy_num, hs) -> (bs, len, enemy_num, fnum+hs)
        enemy_feature = torch.cat([enemy_feature, enemy_buff_maxpool], 3)

        # (bs, len, enemy_num, all_feat_num) -> (bs, len, enemy_num, hs)
        enemy_hidden = self.activator(self.enemy_fc(enemy_feature))
        enemy_hidden = nn.LayerNorm(enemy_hidden.size()[2:], elementwise_affine=False)(enemy_hidden)
        # (bs, len, enemy_num, hs) -> (bs, len, hs)
        # filter invalid enemy info by learned fc and maxpooling, whether explicit mask is needed ?
        enemy_hidden_maxpool, _ = torch.max(enemy_hidden, 2)

        """ task 1: skill select """
        # concate
        concat_hidden = torch.cat([mp_hidden, ally_hidden_maxpool, enemy_hidden_maxpool], 2)

        logger.debug("======= 2.build rnn net here =======")
        # Preprocess observation with FC
        lstm_out = self.activator(self.fc(concat_hidden))
        lstm_out = nn.LayerNorm(lstm_out.size()[1:], elementwise_affine=False)(lstm_out)

        # compute skill logits  (bs, len, skill_num)
        skill_logits = self.skill_dense(lstm_out)

        """ task 2: target select """
        skill_feature = torch.stack(torch.split(skill_feature, [SKILL_FEAT_NUM for _ in range(ACTION_NUM)], -1), 2)

        # skill mask
        # (bs, len, skill_num) * (bs, len, skill_num)-> (bs, len, skill_num)
        skill_logits = skill_logits * skill_mask + torch.log(torch.tensor(self.eps))

        # skill embedding
        # (bs, len, skill_num) -> (bs, len)
        skill_slot = skill_slot_label
        skill_slot_pred = torch.argmax(skill_logits, -1)

        skill_target_op = self._skill_target_op
        
        # use label skill to select
        ally_selected_logist, enemy_selected_logist, skill_t_ally_mask, skill_t_enemy_mask = \
            skill_target_op(skill_slot, hero_id, skill_feature, mp_hidden, ally_hidden, enemy_hidden,
                                  ally_mask, enemy_mask, skill_t_ally_masks, skill_t_enemy_masks, concat_hidden)
        # use predict skill to select
        ally_selected_logist_inference, enemy_selected_logist_inference, skill_t_ally_mask_inference, skill_t_enemy_mask_inference = \
            skill_target_op(skill_slot_pred, hero_id, skill_feature, mp_hidden, ally_hidden, enemy_hidden,
                                  ally_mask, enemy_mask, skill_t_ally_masks, skill_t_enemy_masks, concat_hidden)

        # policy concate logits
        # [bs, len, skill_num], [bs, len, ally_num], [bs, len, enemy_num] -> [bs, len, skill_num+ally_num+enemy_num]
        concat_logits = torch.cat([skill_logits, ally_selected_logist, enemy_selected_logist], -1)
        concat_logits_inference = torch.cat([skill_logits, ally_selected_logist_inference, enemy_selected_logist_inference], -1)

        """ output """
        values = self.value_dense(lstm_out)

        # self.unit_test()

        # plot gragh, must use disable_eager_execution
        # tf.keras.utils.plot_model(self.model, show_shapes=True,show_dtype=True,rankdir="LR", to_file=gOutputPath)
        # print("plot end")
       
        return concat_logits, concat_logits_inference, values, skill_mask, skill_t_ally_mask, skill_t_enemy_mask, skill_t_ally_mask_inference, skill_t_enemy_mask_inference, debug
    
    def forward(self, padded_inputs, state=None, seq_lens=None):
        # Assuming you have a logger defined
        logger.debug("======= 1.define custom forward policy/value net here =======")

        # (bs, dim) - >(bs, lem=1, dim)
        seq_inputs = padded_inputs.unsqueeze(1)

        # Apply the model to seq_inputs (assuming you have defined the layers and connections in the __init__ method and other helper methods)
        concat_logits, concat_logits_inference, self._value_out, skill_mask, skill_t_ally_mask, skill_t_enemy_mask, skill_t_ally_mask_inference, skill_t_enemy_mask_inference, debug = self.model(seq_inputs)

        if self.mode != "sl_train":
            concat_logits, skill_t_ally_mask, skill_t_enemy_mask = concat_logits_inference, skill_t_ally_mask_inference, skill_t_enemy_mask_inference

        # (bs, len, dim) -> (bs, dim)
        concat_logits = concat_logits.view(-1, self.num_outputs)
        # (bs, len, 3) -> (bs, 3)
        skill_mask = skill_mask.view(-1, 3)
        # (bs, len, 1) -> (bs)
        skill_t_ally_mask = skill_t_ally_mask.view(-1)
        skill_t_enemy_mask = skill_t_enemy_mask.view(-1)
        
        return concat_logits, skill_mask, skill_t_ally_mask, skill_t_enemy_mask
    
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
    
    def value_function(self):
        logger.debug("======= 1.define value_out here =======")
        return self._value_out.view(-1)
    
    def custom_loss(self, policy_loss, loss_inputs):
        self.policy_loss = policy_loss
        self.imitation_loss = policy_loss 
        return policy_loss
    
    def metrics(self):
        return {
            #"policy_loss": self.policy_loss,
            #"imitation_loss": self.imitation_loss,
        }
    
    def unit_test(self):
        print("========================= unit test ========================")
        mlen = 5
        seq_inputs = torch.ones((10, mlen, 1960), dtype=torch.float32)
        seq_lens = torch.ones((10)) * mlen
        state_c = torch.ones((10, self.cell_size), dtype=torch.float32)
        state_h = torch.ones((10, self.cell_size), dtype=torch.float32)

        inputs = [seq_inputs, seq_lens, state_c, state_h]
        
        model_out, self._value_out, h, c = self(inputs)
        print("===" * 6)
