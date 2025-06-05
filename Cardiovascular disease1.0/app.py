from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

# 初始化Flask应用
app = Flask(__name__)  # 指定静态文件目录为'static'

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardio_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 定义年龄分层
AGE_BINS = [0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
AGE_LABELS = ['0-20', '20-30', '30-35', '35-40', '40-45', '45-50', 
              '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', 
              '80-85', '85-90', '90-95', '95-100']

# 定义BMI分层
BMI_BINS = [0, 18.5, 23.9, 27.9, 49.9]
BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese']
BMI_CODES = [0, 1, 2, 3]  # 修改为数值代码，与聚类数据匹配

# 定义MAP分层
MAP_BINS = [0, 69.9, 104.9, 119.9, float('inf')]
MAP_LABELS = ['Low', 'Normal', 'High', 'Very High']
MAP_CODES = [0, 0, 1, 2]  # 修改为数值代码，与聚类数据匹配

# 簇类描述映射
CLUSTER_DESCRIPTIONS = {
    0: "女性高风险组",
    1: "女性低风险组",
    2: "男性低风险组",
    3: "男性高风险组"
}

def get_cluster(gender, age_bin_code, BMI_Code, MAP_Code, cholesterol, gluc, smoke):
    """
    使用模糊逻辑对用户进行聚类，避免无法分类的情况
    
    参数:
        gender: 性别 ('女' 或 '男')
        age_bin_code: 年龄分组代码 (0-15)
        BMI_Code: BMI等级代码 (0-4)
        MAP_Code: 血压等级代码 (0-3)
        cholesterol: 胆固醇水平 (0-2)
        gluc: 血糖水平 (0-2)
        smoke: 是否吸烟 (0=否, 1=是)
    
    返回:
        簇类编号 (0-3) 及对应的描述
    """
    # 定义各特征的风险权重
    gender_weight = 1.0
    age_weight = 0.8
    bmi_weight = 0.7
    map_weight = 0.9
    cholesterol_weight = 0.8
    gluc_weight = 0.8
    smoke_weight = 1.0  # 吸烟权重较高
    
    # 初始化各簇的得分
    scores = [0, 0, 0, 0]
    
    # 修改后（女性高风险=0，低风险=1）
    if gender == '女':
        scores[0] += 1.0 * gender_weight  # 女性高风险组（簇0）
        scores[1] += 1.0 * gender_weight  # 女性低风险组（簇1）
    else:
        scores[2] += 1.0 * gender_weight  # 男性低风险组（簇2）
        scores[3] += 1.0 * gender_weight  # 男性高风险组（簇3）
    
    # 年龄特征对簇的贡献 - 越年轻越倾向低风险
    age_risk = age_bin_code / max(AGE_BINS)
    scores[1] += (1 - age_risk) * age_weight  # 女性低风险组
    scores[0] += age_risk * age_weight        # 女性高风险组
    scores[2] += (1 - age_risk) * age_weight  # 男性低风险组
    scores[3] += age_risk * age_weight        # 男性高风险组
    
    # BMI特征对簇的贡献 - BMI越高越倾向高风险
    bmi_risk = min(BMI_Code / max(BMI_CODES), 1.0)
    scores[1] += (1 - bmi_risk) * bmi_weight  # 女性低风险组
    scores[0] += bmi_risk * bmi_weight        # 女性高风险组
    scores[2] += (1 - bmi_risk) * bmi_weight  # 男性低风险组
    scores[3] += bmi_risk * bmi_weight        # 男性高风险组
    
    # MAP特征对簇的贡献 - MAP越高越倾向高风险
    map_risk = min(MAP_Code / max(MAP_CODES), 1.0)
    scores[1] += (1 - map_risk) * map_weight  # 女性低风险组
    scores[0] += map_risk * map_weight        # 女性高风险组
    scores[2] += (1 - map_risk) * map_weight  # 男性低风险组
    scores[3] += map_risk * map_weight        # 男性高风险组
    
    # 胆固醇特征对簇的贡献
    cholesterol_risk = cholesterol / 2.0
    scores[1] += (1 - cholesterol_risk) * cholesterol_weight  # 女性低风险组
    scores[0] += cholesterol_risk * cholesterol_weight        # 女性高风险组
    scores[2] += (1 - cholesterol_risk) * cholesterol_weight  # 男性低风险组
    scores[3] += cholesterol_risk * cholesterol_weight        # 男性高风险组
    
    # 血糖特征对簇的贡献
    gluc_risk = gluc / 2.0
    scores[1] += (1 - gluc_risk) * gluc_weight  # 女性低风险组
    scores[0] += gluc_risk * gluc_weight        # 女性高风险组
    scores[2] += (1 - gluc_risk) * gluc_weight  # 男性低风险组
    scores[3] += gluc_risk * gluc_weight        # 男性高风险组
    
    # 吸烟特征对簇的贡献
    if smoke == 1:
        scores[1] -= 0.5 * smoke_weight  # 降低女性低风险组得分
        scores[0] += 0.5 * smoke_weight  # 增加女性高风险组得分
        scores[2] -= 0.5 * smoke_weight  # 降低男性低风险组得分
        scores[3] += 0.5 * smoke_weight  # 增加男性高风险组得分
    
    # 确定最终簇类 - 选择得分最高的簇
    cluster_id = scores.index(max(scores))
    
    # 计算确定性得分 (0-100%)
    confidence = max(scores) / sum(scores) * 100 if sum(scores) > 0 else 0
    
    # 添加描述信息，包括确定性
    description = f"{CLUSTER_DESCRIPTIONS[cluster_id]} (确定性: {confidence:.1f}%)"
    
    return cluster_id, description

# 创建编码器映射
encoders = {
    'age_bin': preprocessing.LabelEncoder(),
    'BMI_Class': preprocessing.LabelEncoder(),
    'MAP_Class': preprocessing.LabelEncoder()
}

# 定义数据模型 - 添加分层字段
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Integer)
    age_bin = db.Column(db.String(10))  # 年龄分层
    gender = db.Column(db.Integer)
    height = db.Column(db.Integer)
    weight = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    BMI_Class = db.Column(db.String(20))  # BMI分层
    BMI_Code = db.Column(db.Integer)      # BMI代码
    ap_hi = db.Column(db.Integer)
    ap_lo = db.Column(db.Integer)
    map = db.Column(db.Float)
    MAP_Class = db.Column(db.String(20))  # MAP分层
    MAP_Code = db.Column(db.Integer)      # MAP代码
    cholesterol = db.Column(db.Integer)
    gluc = db.Column(db.Integer)
    smoke = db.Column(db.Integer)
    alco = db.Column(db.Integer)
    active = db.Column(db.Integer)
    prediction = db.Column(db.Float)
    cluster = db.Column(db.Integer)       # 健康聚类
    cluster_description = db.Column(db.String(50))  # 新增：簇类描述
    cluster_confidence = db.Column(db.Float)  # 新增：聚类确定性

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'age': self.age,
            'age_bin': self.age_bin,
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'bmi': self.bmi,
            'BMI_Class': self.BMI_Class,
            'BMI_Code': self.BMI_Code,
            'ap_hi': self.ap_hi,
            'ap_lo': self.ap_lo,
            'map': self.map,
            'MAP_Class': self.MAP_Class,
            'MAP_Code': self.MAP_Code,
            'cholesterol': self.cholesterol,
            'gluc': self.gluc,
            'smoke': self.smoke,
            'alco': self.alco,
            'active': self.active,
            'prediction': self.prediction,
            'cluster': self.cluster,
            'cluster_description': self.cluster_description,
            'cluster_confidence': self.cluster_confidence  
        }

# 创建数据库表
with app.app_context():
    db.create_all()

# 加载模型
model_path = 'best_model_random_forest.pkl'  # 请确保模型文件在正确路径下
try:
    model = joblib.load(model_path)
    print(f"模型已成功加载: {model_path}")
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None

# 主页路由 - 显示表单
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if model is None:
                return render_template('index.html', error='模型未加载，请检查模型路径')

            # 获取表单数据
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            height = int(request.form['height'])
            weight = int(request.form['weight'])
            ap_hi = int(request.form['ap_hi'])
            ap_lo = int(request.form['ap_lo'])
            cholesterol = int(request.form['cholesterol'])
            gluc = int(request.form['gluc'])
            smoke = int(request.form['smoke'])
            alco = int(request.form['alco'])
            active = int(request.form['active'])

            # 计算BMI和MAP
            height_meters = height / 100  # 将厘米转换为米
            bmi = weight / (height_meters ** 2)  # BMI计算
            map_value = (ap_hi + 2 * ap_lo) / 3  # MAP计算
            
            # 获取分层信息
            age_bin = get_age_group(age)
            BMI_Class, BMI_Code = get_bmi_info(bmi)
            MAP_Class, MAP_Code = get_map_info(map_value)
            
            # 获取年龄分层代码（用于聚类判断）
            age_bin_code = AGE_LABELS.index(age_bin) if age_bin in AGE_LABELS else 0
            
            # 将性别转换为字符串
            gender_str = '女' if gender == 1 else '男'
            
            # 获取聚类结果
            cluster_id, cluster_description = get_cluster(
                gender=gender_str,
                age_bin_code=age_bin_code,
                BMI_Code=BMI_Code,
                MAP_Code=MAP_Code,
                cholesterol=cholesterol,
                gluc=gluc,
                smoke=smoke
            )
            
            # 提取聚类确定性
            confidence_start = cluster_description.find('(确定性:')
            confidence_end = cluster_description.find('%)')
            confidence = float(cluster_description[confidence_start+5:confidence_end].strip())
            
            # 准备模型输入
            input_data = pd.DataFrame([{
                'Cluster': cluster_id,
                'gender': gender,
                'age_bin': age_bin_code,
                'BMI_Class': BMI_Code,
                'MAP_Class': MAP_Code,
                'cholesterol': cholesterol,
                'gluc': gluc,
                'smoke': smoke,
                'active': active
            }])

            # 使用模型进行预测
            prediction = model.predict_proba(input_data)[:, 1][0]  # 获取阳性类别的概率
            risk_level = get_risk_level(prediction)
            recommendations = get_recommendations(
                {
                    'age': age, 'gender': gender, 'height': height, 'weight': weight,
                    'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
                    'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
                },
                prediction, bmi, map_value, age_bin, BMI_Class, BMI_Code, MAP_Class, MAP_Code, cluster_id, confidence
            )
            risk_factors = get_risk_factors(
                {
                    'age': age, 'gender': gender, 'height': height, 'weight': weight,
                    'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
                    'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
                },
                prediction, bmi, map_value, age_bin, BMI_Class, BMI_Code, MAP_Class, MAP_Code, cluster_id, confidence
            )

            # 保存到数据库
            new_prediction = Prediction(
                age=age,
                age_bin=age_bin,
                gender=gender,
                height=height,
                weight=weight,
                bmi=bmi,
                BMI_Class=BMI_Class,
                BMI_Code=BMI_Code,
                ap_hi=ap_hi,
                ap_lo=ap_lo,
                map=map_value,
                MAP_Class=MAP_Class,
                MAP_Code=MAP_Code,
                cholesterol=cholesterol,
                gluc=gluc,
                smoke=smoke,
                alco=alco,
                active=active,
                prediction=prediction,
                cluster=cluster_id,
                cluster_description=cluster_description,
                cluster_confidence=confidence  # 保存聚类确定性
            )
            db.session.add(new_prediction)
            db.session.commit()

            # 将结果传递给模板
            return render_template(
                'index.html',
                prediction=prediction,
                risk_level=risk_level,
                recommendations=recommendations,
                risk_factors=risk_factors,
                bmi=bmi,
                map=map_value,
                age_bin=age_bin,
                BMI_Class=BMI_Class,
                BMI_Code=BMI_Code,
                MAP_Class=MAP_Class,
                MAP_Code=MAP_Code,
                cluster=cluster_id,
                cluster_description=cluster_description,
                cluster_confidence=confidence,  # 传递给模板
                form_data=request.form
            )

        except Exception as e:
            return render_template('index.html', error=str(e))
    
    # GET请求 - 返回空表单
    return render_template('index.html')

# 将年龄转换为分层标签
def get_age_group(age):
    for i, (lower, upper) in enumerate(zip(AGE_BINS[:-1], AGE_BINS[1:])):
        if lower <= age < upper:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]  # 如果年龄大于等于100

# 将BMI转换为分层标签和代码
def get_bmi_info(bmi):
    for i, (lower, upper) in enumerate(zip(BMI_BINS[:-1], BMI_BINS[1:])):
        if lower <= bmi < upper:
            return BMI_LABELS[i], BMI_CODES[i]
    return BMI_LABELS[-1], BMI_CODES[-1]  # 如果BMI超出范围

# 将MAP转换为分层标签和代码
def get_map_info(map_value):
    for i, (lower, upper) in enumerate(zip(MAP_BINS[:-1], MAP_BINS[1:])):
        if lower <= map_value < upper:
            return MAP_LABELS[i], MAP_CODES[i]
    return MAP_LABELS[-1], MAP_CODES[-1]  # 如果MAP超出范围

# 获取风险等级描述
def get_risk_level(probability):
    if probability < 0.3:
        return '低风险'
    elif probability < 0.7:
        return '中等风险'
    else:
        return '高风险'

# 根据数据和预测结果生成建议 - 添加分层参数
def get_recommendations(data, prediction, bmi, map_value, age_bin, BMI_Class, BMI_Code, MAP_Class, MAP_Code, cluster, confidence):
    recommendations = []
    
    if prediction >= 0.6:
        recommendations.append('您的心血管疾病风险较高，请立即咨询专业医生进行进一步检查。')
    
    # 基于BMI的建议
    if BMI_Code == 0:  # Underweight
        recommendations.append(f'您的BMI分类为{BMI_Class}，建议增加营养摄入，保持健康体重。')
    elif BMI_Code == 2:  # Overweight
        recommendations.append(f'您的BMI分类为{BMI_Class}，建议控制饮食，减少高热量食物摄入，增加有氧运动。')
    elif BMI_Code == 3:  # Obese
        recommendations.append(f'您的BMI分类为{BMI_Class}，建议严格控制饮食，增加体育活动，并考虑咨询营养师或医生制定减重计划。')
    elif BMI_Code == 4:  # Error
        recommendations.append(f'您的BMI值异常高，建议立即咨询医生进行进一步评估。')
    
    # 基于MAP的建议
    if MAP_Code == 2:  # High
        recommendations.append(f'您的平均动脉压分类为{MAP_Class}，建议减少盐的摄入，增加体育活动，并定期监测血压。')
    elif MAP_Code == 3:  # Very High
        recommendations.append(f'您的平均动脉压分类为{MAP_Class}，这是一个严重的健康问题，建议立即咨询医生并遵循降压治疗方案。')
    elif MAP_Code == 0:  # Low
        recommendations.append(f'您的平均动脉压分类为{MAP_Class}，如果您有头晕或乏力等症状，请咨询医生。')
    
    if data['cholesterol'] > 1:
        recommendations.append('您的胆固醇水平偏高，建议减少饱和脂肪和胆固醇的摄入，增加膳食纤维。')
    
    if data['gluc'] > 1:
        recommendations.append('您的血糖水平偏高，建议控制碳水化合物摄入，增加体育活动，并定期监测血糖。')
    
    # 基于吸烟的建议
    if data['smoke'] == 1:
        recommendations.append('吸烟是心血管疾病的重要危险因素，建议戒烟以降低风险。')
    
    if data['alco'] == 1 and prediction > 0.3:
        recommendations.append('过量饮酒会增加心血管疾病风险，建议适量饮酒或戒酒。')
    
    if data['active'] == 0:
        # 根据年龄分层提供不同的运动建议
        age_lower = int(age_bin.split('-')[0])
        if age_lower < 60:
            recommendations.append('增加体育活动可以显著降低心血管疾病风险，建议每周进行至少150分钟的中等强度有氧运动。')
        else:
            recommendations.append('适量的体育活动如散步、太极拳等对心血管健康有益，建议每天保持一定的活动量。')
    
    # 根据年龄分层提供额外建议
    age_lower = int(age_bin.split('-')[0])
    if age_lower >= 50:
        recommendations.append('随着年龄增长，心血管疾病风险增加，建议定期进行心脏健康检查。')
    
    # 根据聚类提供特定建议
    if cluster == 1 or cluster == 3:  # 高风险簇
        recommendations.append(f'根据您的健康指标聚类分析结果（确定性{confidence:.1f}%），您属于高风险组，建议制定个性化健康管理计划并定期复查。')
    elif cluster == 0 or cluster == 2:  # 低风险簇
        recommendations.append(f'根据您的健康指标聚类分析结果（确定性{confidence:.1f}%），您属于低风险组，建议保持健康生活方式，定期体检。')
    
    # 如果确定性较低，建议重新评估
    if confidence < 70:
        recommendations.append(f'您的聚类分析结果确定性为{confidence:.1f}%，建议定期重新评估健康状况，确保评估准确性。')
    
    if not recommendations:
        recommendations = [
            '继续保持健康的生活方式，定期锻炼。',
            '保持均衡饮食，减少饱和脂肪和盐的摄入。',
            '定期体检，监测血压和胆固醇水平。'
        ]
    
    return recommendations

# 生成风险因素分析 - 添加分层参数
def get_risk_factors(data, prediction, bmi, map_value, age_bin, BMI_Class, BMI_Code, MAP_Class, MAP_Code, cluster, confidence):
    risk_factors = []
    
    # 基于BMI的风险因素
    if BMI_Code == 1:  # Normal
        risk_factors.append(f'您的BMI分类为{BMI_Class}，保持健康的体重对心血管健康很重要。')
    elif BMI_Code == 4:
        risk_factors.append(f'您的BMI值异常高，这是心血管疾病的重大危险因素。')
    else:
        risk_factors.append(f'您的BMI分类为{BMI_Class}，这可能会增加心血管疾病风险。')
    
    # 基于MAP的风险因素
    if MAP_Code == 1:  # Normal
        risk_factors.append(f'您的平均动脉压分类为{MAP_Class}，这对心血管健康非常有益。')
    elif MAP_Code == 3:  # Very High
        risk_factors.append(f'您的平均动脉压分类为{MAP_Class}，这是心血管疾病的严重危险因素，需要紧急医疗干预。')
    else:
        risk_factors.append(f'您的平均动脉压分类为{MAP_Class}，这是心血管疾病的重要危险因素。')
    
    if data['cholesterol'] == 0:
        risk_factors.append('您的胆固醇水平正常，建议继续保持。')
    elif data['cholesterol'] > 1:
        risk_factors.append('您的胆固醇水平偏高，高胆固醇是心血管疾病的主要危险因素之一。')
    
    if data['gluc'] == 0:
        risk_factors.append('您的血糖水平正常，继续保持健康的饮食习惯。')
    elif data['gluc'] > 1:
        risk_factors.append('您的血糖水平偏高，高血糖会增加心血管疾病风险。')
    
    # 基于吸烟的风险因素
    if data['smoke'] == 0:
        risk_factors.append('您不吸烟，这是心血管健康的重要保护因素。')
    else:
        risk_factors.append('吸烟是心血管疾病的重要危险因素，戒烟可以显著降低风险。')
    
    if data['active'] == 1:
        risk_factors.append('您的体育活动水平有助于降低心血管疾病风险。')
    else:
        risk_factors.append('缺乏体育活动会增加心血管疾病风险，建议增加日常活动量。')
    
    # 添加年龄分层风险因素
    age_lower = int(age_bin.split('-')[0])
    if age_lower >= 50:
        risk_factors.append(f'您属于{age_bin}年龄段，心血管疾病风险随年龄增长而增加。')
    elif age_lower >= 30:
        risk_factors.append(f'您属于{age_bin}年龄段，建议开始关注心血管健康，保持健康生活方式。')
    else:
        risk_factors.append(f'您属于{age_bin}年龄段，保持当前健康生活方式可有效预防心血管疾病。')
    
    # 综合风险因素
    if prediction >= 0.6:
        risk_factors.append(f'基于您的各项指标，您的心血管疾病风险评估为高风险({prediction:.1%})，请立即咨询医生。')
    elif prediction >= 0.3:
        risk_factors.append(f'基于您的各项指标，您的心血管疾病风险评估为中等风险({prediction:.1%})，建议改善生活习惯并定期体检。')
    else:
        risk_factors.append(f'基于您的各项指标，您的心血管疾病风险评估为低风险({prediction:.1%})，请继续保持健康生活方式。')
    
    # 添加聚类风险因素
    if cluster in CLUSTER_DESCRIPTIONS:
        risk_factors.append(f'根据健康指标聚类分析（确定性{confidence:.1f}%），您属于健康聚类{cluster}：{CLUSTER_DESCRIPTIONS[cluster]}')
    else:
        risk_factors.append(f'根据健康指标聚类分析，您的健康状况无法明确归类。')
    
    return risk_factors

# API路由 - 获取所有预测记录
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        predictions = Prediction.query.all()
        return jsonify([p.to_dict() for p in predictions]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """显示历史预测结果页面"""
    # 查询所有预测记录，按时间倒序排列（最新的在前）
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    
    # 将预测结果转换为可展示的格式
    formatted_predictions = []
    for pred in predictions:
        # 格式化日期时间
        timestamp_str = pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # 根据簇类ID获取描述
        cluster_desc = CLUSTER_DESCRIPTIONS.get(pred.cluster, "未知类别")
        
        # 格式化风险等级
        risk_level = get_risk_level(pred.prediction)
        
        formatted_predictions.append({
            'id': pred.id,
            'timestamp': timestamp_str,
            'age': pred.age,
            'gender': '女' if pred.gender == 1 else '男',
            'bmi': round(pred.bmi, 1),
            'ap_hi': pred.ap_hi,
            'ap_lo': pred.ap_lo,
            'prediction': round(pred.prediction * 100, 1),  # 转换为百分比
            'risk_level': risk_level,
            'cluster_description': cluster_desc
        })
    
    return render_template('history.html', predictions=formatted_predictions)

@app.route('/history/<int:prediction_id>')
def prediction_detail(prediction_id):
    """显示单个预测结果的详细信息"""
    pred = Prediction.query.get_or_404(prediction_id)
    
    # 格式化日期时间
    timestamp_str = pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # 根据簇类ID获取描述
    cluster_desc = CLUSTER_DESCRIPTIONS.get(pred.cluster, "未知类别")
    
    # 格式化风险等级
    risk_level = get_risk_level(pred.prediction)
    
    # 修复 age_bin 类型错误
    try:
        age_bin_index = AGE_LABELS.index(pred.age_bin)
    except ValueError:
        age_bin_index = 0
    
    # 生成健康建议和风险因素
    recommendations = get_recommendations(
        {
            'age': pred.age, 'gender': pred.gender, 'height': pred.height, 'weight': pred.weight,
            'ap_hi': pred.ap_hi, 'ap_lo': pred.ap_lo, 'cholesterol': pred.cholesterol,
            'gluc': pred.gluc, 'smoke': pred.smoke, 'alco': pred.alco, 'active': pred.active
        },
        pred.prediction, pred.bmi, pred.map, 
        AGE_LABELS[age_bin_index],
        BMI_LABELS[min(pred.BMI_Code, len(BMI_LABELS)-1)], 
        pred.BMI_Code, 
        MAP_LABELS[min(pred.MAP_Code, len(MAP_LABELS)-1)], 
        pred.MAP_Code, 
        pred.cluster, 
        pred.cluster_confidence
    )
    
    risk_factors = get_risk_factors(
        {
            'age': pred.age, 'gender': pred.gender, 'height': pred.height, 'weight': pred.weight,
            'ap_hi': pred.ap_hi, 'ap_lo': pred.ap_lo, 'cholesterol': pred.cholesterol,
            'gluc': pred.gluc, 'smoke': pred.smoke, 'alco': pred.alco, 'active': pred.active
        },
        pred.prediction, pred.bmi, pred.map, 
        AGE_LABELS[age_bin_index],
        BMI_LABELS[min(pred.BMI_Code, len(BMI_LABELS)-1)], 
        pred.BMI_Code, 
        MAP_LABELS[min(pred.MAP_Code, len(MAP_LABELS)-1)], 
        pred.MAP_Code, 
        pred.cluster, 
        pred.cluster_confidence
    )
    
    # 将必要的常量传递到模板
    return render_template('prediction_detail.html', 
                          prediction=pred,
                          timestamp=timestamp_str,
                          cluster_desc=cluster_desc,
                          risk_level=risk_level,
                          recommendations=recommendations,
                          risk_factors=risk_factors,
                          AGE_LABELS=AGE_LABELS,
                          BMI_LABELS=BMI_LABELS,
                          MAP_LABELS=MAP_LABELS,
                          CLUSTER_DESCRIPTIONS=CLUSTER_DESCRIPTIONS)

if __name__ == '__main__':
    app.run(debug=True)