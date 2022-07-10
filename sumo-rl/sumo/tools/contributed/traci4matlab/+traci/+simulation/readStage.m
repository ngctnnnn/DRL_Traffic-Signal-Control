function stage = readStage(result)

result.readInt();
result.read(1);
stageType = result.readInt();
result.read(1);
vType = result.readString();
result.read(1);
line = result.readString();
result.read(1);
destStop = result.readString();
result.read(1);
edges = result.readStringList();
result.read(1);
travelTime = result.readDouble();
result.read(1);
cost = result.readDouble();
result.read(1);
result.read(1);
length = result.readDouble();
intended = result.readString();
result.read(1);
depart = result.readDouble();
result.read(1);
departPos = result.readDouble();
result.read(1);
arrivalPos = result.readDouble();
result.read(1);
description = result.readString();

stage = struct('stageType', stageType, 'vType', vType, 'line', line,...
    'destStop', destStop, 'edges', edges, 'travelTime', travelTime,...
    'cost', cost, 'length', length, 'intended', intended, 'depart',...
    depart, 'departPos', departPos, 'arrivalPos', arrivalPos,...
    'description', description);
% stage.edges = edges;
% stage.travelTime = travelTime;
% stage.cost = cost;
% stage.intended = intended;
% stage.depart = depart;
